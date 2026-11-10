package com.dsrnative.app;

import android.app.Activity;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.hardware.input.InputManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import java.util.ArrayList;
import java.util.List;

/**
 * On-screen controls drawn in a transparent panel window over the game. A fixed stick at the bottom left
 * moves, a smaller stick appears wherever the right side of the screen is touched (not on a button) and turns
 * the camera, and the Switch buttons sit around the edges.
 * They hide when a game controller is connected or used, and come back with a double tap on the screen.
 * State goes to native code (nativeState), which merges it with the controller.
 */
public final class TouchControls extends View implements InputManager.InputDeviceListener {
    // nn::hid button bits (src/hle/input.h).
    static final int A = 1, B = 1 << 1, X = 1 << 2, Y = 1 << 3, L3 = 1 << 4, R3 = 1 << 5, L = 1 << 6, R = 1 << 7,
            ZL = 1 << 8, ZR = 1 << 9, PLUS = 1 << 10, MINUS = 1 << 11, LEFT = 1 << 12, UP = 1 << 13, RIGHT = 1 << 14,
            DOWN = 1 << 15;

    static native void nativeState(int buttons, float lx, float ly, float rx, float ry);
    static native void nativeVisible(boolean visible);

    private static TouchControls sView;

    /** Called by native code once the game starts. */
    public static void attach(final Activity activity, final float opacity, final float cameraSpeed,
            final boolean enabled) {
        activity.runOnUiThread(() -> {
            if (sView != null || !enabled) return;
            final TouchControls v = new TouchControls(activity, opacity, cameraSpeed);
            v.addTo(activity, 0);
        });
    }

    /** Called by native code when a controller is used while the controls show. */
    public static void controllerUsed(final Activity activity) {
        activity.runOnUiThread(() -> {
            if (sView != null) sView.setShown(false);
        });
    }

    private final Activity mActivity;
    private final float mDensity;
    private final Handler mHandler = new Handler(Looper.getMainLooper());
    private boolean mShown = true;

    private static final int K_BUTTON = 0, K_STICK = 1, K_RSTICK = 2, K_NONE = 3;

    private static final class Button {
        final String label;
        final int bit;
        final boolean round;
        float cx, cy, w, h;  // centre and size (px)
        int pressCount;
        Button(String label, int bit, boolean round) {
            this.label = label;
            this.bit = bit;
            this.round = round;
        }
        boolean hit(float x, float y, float slop) {
            if (round) {
                float r = w / 2 + slop, dx = x - cx, dy = y - cy;
                return dx * dx + dy * dy <= r * r;
            }
            return Math.abs(x - cx) <= w / 2 + slop && Math.abs(y - cy) <= h / 2 + slop;
        }
    }

    private final List<Button> mButtons = new ArrayList<>();

    // Per finger (pointer id 0..15): what it does.
    private final int[] mKind = new int[16];
    private final Button[] mOn = new Button[16];
    private final float[] mLastX = new float[16], mLastY = new float[16];

    // Stick: fixed in place; a finger that comes down on it (or near it) drives it.
    private int mStickPointer = -1;
    private float mStickBaseX, mStickBaseY, mStickX, mStickY, mStickRadius;
    // Camera stick: invisible until the right side is touched; appears under the finger and its base follows a
    // finger that goes past its edge. Smaller than the left stick; `camera speed` = deflection per distance.
    private int mRStickPointer = -1;
    private float mRBaseX, mRBaseY, mRX, mRY, mRRadius;
    // A thumb resting on it (not moving for a moment): the base slides under the thumb, so the camera eases to
    // a stop instead of turning on and on.
    private long mRLastMove;
    private long mRLastTick;
    private static final long REST_DELAY_MS = 80;
    private static final float RECENTER_TIME_S = 0.35f;  // time constant of the slide
    private final Runnable mRecenter = new Runnable() {
        @Override
        public void run() {
            if (mRStickPointer < 0) return;
            long now = android.os.SystemClock.uptimeMillis();
            float dt = (now - mRLastTick) / 1000f;
            mRLastTick = now;
            if (now - mRLastMove >= REST_DELAY_MS && (mRX != mRBaseX || mRY != mRBaseY)) {
                float k = 1f - (float) Math.exp(-dt / RECENTER_TIME_S);
                mRBaseX += (mRX - mRBaseX) * k;
                mRBaseY += (mRY - mRBaseY) * k;
                float dx = mRX - mRBaseX, dy = mRY - mRBaseY;
                if (dx * dx + dy * dy < 1f) {
                    mRBaseX = mRX;
                    mRBaseY = mRY;
                }
                sendState();
                invalidate();
            }
            mHandler.postDelayed(this, 16);
        }
    };
    private final float mCameraSpeed;

    // Double tap (to show the controls again).
    private long mLastTapTime;
    private float mLastTapX, mLastTapY;

    private final Paint mFill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mStroke = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF mRect = new RectF();

    private TouchControls(Activity activity, float opacity, float cameraSpeed) {
        super(activity);
        mActivity = activity;
        mCameraSpeed = Math.max(0.1f, Math.min(10f, cameraSpeed));
        mDensity = activity.getResources().getDisplayMetrics().density;
        setAlpha(Math.max(0.1f, Math.min(1.0f, opacity)));
        mStroke.setStyle(Paint.Style.STROKE);
        mStroke.setColor(0xFFFFFFFF);
        mText.setColor(0xFFFFFFFF);
        mText.setTextAlign(Paint.Align.CENTER);
        mText.setFakeBoldText(true);
        mButtons.add(new Button("A", A, true));
        mButtons.add(new Button("B", B, true));
        mButtons.add(new Button("X", X, true));
        mButtons.add(new Button("Y", Y, true));
        mButtons.add(new Button("L", L, false));
        mButtons.add(new Button("ZL", ZL, false));
        mButtons.add(new Button("R", R, false));
        mButtons.add(new Button("ZR", ZR, false));
        mButtons.add(new Button("−", MINUS, true));
        mButtons.add(new Button("+", PLUS, true));
        mButtons.add(new Button("▲", UP, false));
        mButtons.add(new Button("▼", DOWN, false));
        mButtons.add(new Button("◀", LEFT, false));
        mButtons.add(new Button("▶", RIGHT, false));
        mButtons.add(new Button("L3", L3, true));
        mButtons.add(new Button("LOCK", R3, true));
        for (int i = 0; i < 16; i++) mKind[i] = K_NONE;
        // A controller already connected (or built in, as on handhelds): start hidden.
        if (controllerConnected()) mShown = false;
        InputManager im = (InputManager) activity.getSystemService(Context.INPUT_SERVICE);
        if (im != null) im.registerInputDeviceListener(this, mHandler);
    }

    private void addTo(final Activity activity, final int attempt) {
        final View decor = activity.getWindow().getDecorView();
        if (decor.getWindowToken() == null) {  // the game's window is not attached yet
            if (attempt < 50) mHandler.postDelayed(() -> addTo(activity, attempt + 1), 100);
            return;
        }
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        lp.token = decor.getWindowToken();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        try {
            activity.getWindowManager().addView(this, lp);
            sView = this;
            nativeVisible(mShown);
        } catch (Exception e) {
            android.util.Log.w("DSRNative", "touch controls: could not add the overlay", e);
        }
    }

    private boolean controllerConnected() {
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice d = InputDevice.getDevice(id);
            if (d == null || d.isVirtual()) continue;
            int s = d.getSources();
            if ((s & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
                    || (s & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK)
                return true;
        }
        return false;
    }

    @Override
    public void onInputDeviceAdded(int deviceId) {
        InputDevice d = InputDevice.getDevice(deviceId);
        if (d == null) return;
        int s = d.getSources();
        if ((s & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
                || (s & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK)
            setShown(false);
    }

    @Override
    public void onInputDeviceRemoved(int deviceId) {}

    @Override
    public void onInputDeviceChanged(int deviceId) {}

    void setShown(boolean shown) {
        if (mShown == shown) return;
        mShown = shown;
        releaseAll();
        nativeVisible(shown);
        invalidate();
    }

    private void releaseAll() {
        for (int i = 0; i < 16; i++) {
            mKind[i] = K_NONE;
            mOn[i] = null;
        }
        for (Button b : mButtons) b.pressCount = 0;
        mStickPointer = -1;
        mRStickPointer = -1;
        sendState();
    }

    // ---- layout ---------------------------------------------------------------------------------------------

    @Override
    protected void onSizeChanged(int w, int h, int ow, int oh) {
        super.onSizeChanged(w, h, ow, oh);
        layout(w, h);
    }

    private Button find(int bit) {
        for (Button b : mButtons)
            if (b.bit == bit) return b;
        return null;
    }

    private void place(int bit, float cx, float cy, float w, float h) {
        Button b = find(bit);
        b.cx = cx;
        b.cy = cy;
        b.w = w;
        b.h = h;
    }

    private void layout(int w, int h) {
        final float u = Math.min(w, h) / 10f;  // layout unit: a tenth of the short side
        // Face buttons, Switch layout (X top, A right, B bottom, Y left).
        final float fcx = w - 3.0f * u, fcy = h - 3.0f * u, d = 1.45f * u, fb = 1.3f * u;
        place(A, fcx + d, fcy, fb, fb);
        place(B, fcx, fcy + d, fb, fb);
        place(X, fcx, fcy - d, fb, fb);
        place(Y, fcx - d, fcy, fb, fb);
        place(R3, w - 5.6f * u, h - 1.2f * u, 1.3f * u, 1.3f * u);
        // Shoulders: ZL / ZR in the top corners, L / R below them.
        final float sw = 2.4f * u, sh = 1.1f * u;
        place(ZL, 1.7f * u, 1.0f * u, sw, sh);
        place(L, 1.7f * u, 2.5f * u, sw, sh);
        place(ZR, w - 1.7f * u, 1.0f * u, sw, sh);
        place(R, w - 1.7f * u, 2.5f * u, sw, sh);
        // Minus / plus at the top centre.
        place(MINUS, w / 2f - 1.3f * u, 0.9f * u, 1.0f * u, 1.0f * u);
        place(PLUS, w / 2f + 1.3f * u, 0.9f * u, 1.0f * u, 1.0f * u);
        // D-pad, bottom, right of the stick.
        final float dcx = 6.9f * u, dcy = h - 2.0f * u, ds = 0.95f * u, db = 0.9f * u;
        place(UP, dcx, dcy - ds, db, db);
        place(DOWN, dcx, dcy + ds, db, db);
        place(LEFT, dcx - ds, dcy, db, db);
        place(RIGHT, dcx + ds, dcy, db, db);
        place(L3, 4.7f * u, h - 0.9f * u, 1.0f * u, 1.0f * u);
        // Stick: resting spot, and the area where a finger grabs it.
        mStickRadius = 1.6f * u;
        mRRadius = 1.05f * u;
        mStickBaseX = 2.8f * u;
        mStickBaseY = h - 3.0f * u;
        mText.setTextSize(0.45f * u);
        mStroke.setStrokeWidth(Math.max(2f, 0.05f * u));
    }

    // ---- touch ----------------------------------------------------------------------------------------------

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        final int action = e.getActionMasked();
        if (!mShown) {
            if (action == MotionEvent.ACTION_DOWN) checkDoubleTap(e.getX(), e.getY());
            return true;
        }
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN: {
                int i = e.getActionIndex();
                down(e.getPointerId(i), e.getX(i), e.getY(i));
                break;
            }
            case MotionEvent.ACTION_MOVE:
                for (int i = 0; i < e.getPointerCount(); i++) move(e.getPointerId(i), e.getX(i), e.getY(i));
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP: {
                int i = e.getActionIndex();
                up(e.getPointerId(i));
                break;
            }
            case MotionEvent.ACTION_CANCEL:
                releaseAll();
                break;
            default:
                break;
        }
        sendState();
        invalidate();
        return true;
    }

    private void checkDoubleTap(float x, float y) {
        long now = android.os.SystemClock.uptimeMillis();
        float slop = 60 * mDensity;
        if (now - mLastTapTime < 350 && Math.abs(x - mLastTapX) < slop && Math.abs(y - mLastTapY) < slop) {
            mLastTapTime = 0;
            setShown(true);
        } else {
            mLastTapTime = now;
            mLastTapX = x;
            mLastTapY = y;
        }
    }

    private Button hit(float x, float y) {
        float slop = 0.15f * Math.min(getWidth(), getHeight()) / 10f;
        Button best = null;
        float bestDist = Float.MAX_VALUE;
        for (Button b : mButtons)
            if (b.hit(x, y, slop)) {
                float dx = x - b.cx, dy = y - b.cy, dist = dx * dx + dy * dy;
                if (dist < bestDist) {
                    bestDist = dist;
                    best = b;
                }
            }
        return best;
    }

    private void down(int id, float x, float y) {
        if (id < 0 || id >= 16) return;
        mLastX[id] = x;
        mLastY[id] = y;
        Button b = hit(x, y);
        if (b != null) {
            mKind[id] = K_BUTTON;
            mOn[id] = b;
            b.pressCount++;
        } else if (mStickPointer < 0 && onStick(x, y)) {
            mKind[id] = K_STICK;
            mStickPointer = id;
            mStickX = x;
            mStickY = y;
        } else if (mRStickPointer < 0 && x > getWidth() * 0.5f) {
            mKind[id] = K_RSTICK;
            mRStickPointer = id;
            mRBaseX = mRX = x;
            mRBaseY = mRY = y;
            mRLastMove = mRLastTick = android.os.SystemClock.uptimeMillis();
            mHandler.removeCallbacks(mRecenter);
            mHandler.postDelayed(mRecenter, 16);
        } else {
            mKind[id] = K_NONE;
        }
    }

    private boolean onStick(float x, float y) {
        float dx = x - mStickBaseX, dy = y - mStickBaseY, r = mStickRadius * 1.5f;
        return dx * dx + dy * dy <= r * r;
    }

    private void move(int id, float x, float y) {
        if (id < 0 || id >= 16) return;
        switch (mKind[id]) {
            case K_BUTTON: {  // slide between buttons (e.g. across the face buttons)
                Button b = hit(x, y);
                if (b != mOn[id]) {
                    if (mOn[id] != null) mOn[id].pressCount--;
                    mOn[id] = b;
                    if (b != null) b.pressCount++;
                }
                break;
            }
            case K_STICK:
                mStickX = x;
                mStickY = y;
                break;
            case K_RSTICK: {
                if (x != mRX || y != mRY) mRLastMove = android.os.SystemClock.uptimeMillis();
                mRX = x;
                mRY = y;
                float dx = x - mRBaseX, dy = y - mRBaseY, len = (float) Math.sqrt(dx * dx + dy * dy);
                if (len > mRRadius) {  // the base follows the finger
                    mRBaseX = x - dx / len * mRRadius;
                    mRBaseY = y - dy / len * mRRadius;
                }
                break;
            }
            default:
                break;
        }
        mLastX[id] = x;
        mLastY[id] = y;
    }

    private void up(int id) {
        if (id < 0 || id >= 16) return;
        if (mKind[id] == K_BUTTON && mOn[id] != null) mOn[id].pressCount--;
        if (mKind[id] == K_STICK) mStickPointer = -1;
        if (mKind[id] == K_RSTICK) mRStickPointer = -1;
        mKind[id] = K_NONE;
        mOn[id] = null;
    }

    private void sendState() {
        int bits = 0;
        for (Button b : mButtons)
            if (b.pressCount > 0) bits |= b.bit;
        float lx = 0, ly = 0;
        if (mStickPointer >= 0) {
            float dx = mStickX - mStickBaseX, dy = mStickY - mStickBaseY;
            float len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len > mStickRadius) {  // past the edge: full tilt in that direction
                dx = dx / len * mStickRadius;
                dy = dy / len * mStickRadius;
            }
            lx = dx / mStickRadius;
            ly = -dy / mStickRadius;
            if (Math.abs(lx) < 0.08f && Math.abs(ly) < 0.08f) lx = ly = 0;
        }
        float rx = 0, ry = 0;
        if (mRStickPointer >= 0) {
            // Full tilt at 1 / speed of the ring's radius (speed 1: at the edge).
            float reach = mRRadius / mCameraSpeed;
            rx = Math.max(-1f, Math.min(1f, (mRX - mRBaseX) / reach));
            ry = Math.max(-1f, Math.min(1f, -(mRY - mRBaseY) / reach));
            if (Math.abs(rx) < 0.06f && Math.abs(ry) < 0.06f) rx = ry = 0;
        }
        nativeState(bits, lx, ly, rx, ry);
    }

    // ---- drawing --------------------------------------------------------------------------------------------

    @Override
    protected void onDraw(Canvas c) {
        if (!mShown) return;
        // Stick (the knob stays inside the ring).
        final float bx = mStickBaseX, by = mStickBaseY;
        mFill.setColor(0x40000000);
        c.drawCircle(bx, by, mStickRadius, mFill);
        c.drawCircle(bx, by, mStickRadius, mStroke);
        float kx = bx, ky = by;
        if (mStickPointer >= 0) {
            float dx = mStickX - bx, dy = mStickY - by, len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len > mStickRadius) {
                dx = dx / len * mStickRadius;
                dy = dy / len * mStickRadius;
            }
            kx = bx + dx;
            ky = by + dy;
        }
        mFill.setColor(mStickPointer >= 0 ? 0xA0FFFFFF : 0x60FFFFFF);
        c.drawCircle(kx, ky, mStickRadius * 0.45f, mFill);
        // Camera stick, only while touched.
        if (mRStickPointer >= 0) {  // half as opaque as the rest
            mFill.setColor(0x20000000);
            c.drawCircle(mRBaseX, mRBaseY, mRRadius, mFill);
            mStroke.setAlpha(0x80);
            c.drawCircle(mRBaseX, mRBaseY, mRRadius, mStroke);
            mStroke.setAlpha(0xFF);
            float dx = mRX - mRBaseX, dy = mRY - mRBaseY, len = (float) Math.sqrt(dx * dx + dy * dy);
            if (len > mRRadius) {
                dx = dx / len * mRRadius;
                dy = dy / len * mRRadius;
            }
            mFill.setColor(0x50FFFFFF);
            c.drawCircle(mRBaseX + dx, mRBaseY + dy, mRRadius * 0.45f, mFill);
        }
        // Buttons.
        for (Button b : mButtons) {
            boolean on = b.pressCount > 0;
            mFill.setColor(on ? 0x90FFFFFF : 0x50000000);
            if (b.round) {
                c.drawCircle(b.cx, b.cy, b.w / 2, mFill);
                c.drawCircle(b.cx, b.cy, b.w / 2, mStroke);
            } else {
                mRect.set(b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2);
                float r = Math.min(b.w, b.h) * 0.3f;
                c.drawRoundRect(mRect, r, r, mFill);
                c.drawRoundRect(mRect, r, r, mStroke);
            }
            mText.setColor(on ? 0xFF000000 : 0xFFFFFFFF);
            float size = mText.getTextSize();
            if (b.label.length() > 2) mText.setTextSize(size * 0.7f);
            c.drawText(b.label, b.cx, b.cy - (mText.descent() + mText.ascent()) / 2, mText);
            mText.setTextSize(size);
        }
    }
}
