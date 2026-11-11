package com.dsrnative.app;

import android.app.Activity;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Build;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Full-screen, landscape, PPSSPP-style "Display Layout" screen: drag/pinch the game's frame directly on
 * the real screen (writes screen_fit=manual, screen_zoom_x/y, screen_offset_x/y to settings.txt) and drag
 * the touch control buttons/stick (reuses TouchControls' own layout editor, standalone -- see
 * TouchControls.createStandalone(), no native "dsr" library loaded here, there is no running game).
 * A top toggle switches which of the two layers is currently touch-interactive; both stay visible.
 */
public class DisplayCalibrationActivity extends Activity {
    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        Window w = getWindow();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            android.view.WindowManager.LayoutParams lp = w.getAttributes();
            lp.layoutInDisplayCutoutMode = android.view.WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            w.setAttributes(lp);
        }
        // Not called here: the decor view's insets controller isn't ready this early and
        // getInsetsController() returns null (confirmed on-device, NullPointerException). onWindowFocusChanged()
        // below calls hideSystemBars() once the window actually has focus, which is also the standard place
        // Android's own immersive-mode guidance puts it.

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);

        // This whole screen is only meaningful in manual fit mode -- persist that immediately so the
        // live drag/pinch below isn't silently ignored by the renderer.
        writeSetting("screen_fit", "manual");

        final ScreenAdjustView screen = new ScreenAdjustView(this);
        root.addView(screen, new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        final TouchControls touch = TouchControls.createStandalone(this);
        touch.attachInline(root);  // added after `screen`: receives touch first when enabled (Buttons mode)

        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setGravity(Gravity.CENTER_VERTICAL);
        FrameLayout.LayoutParams topBarLp = new FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP | Gravity.CENTER_HORIZONTAL);
        topBarLp.topMargin = dp(12);

        final Button screenBtn = new Button(this), buttonsBtn = new Button(this), resetScreenBtn = new Button(this),
                resetButtonsBtn = new Button(this), doneBtn = new Button(this);
        screenBtn.setText("Screen");
        buttonsBtn.setText("Buttons");
        resetScreenBtn.setText("Reset Screen");
        resetButtonsBtn.setText("Reset Buttons");
        doneBtn.setText("Done");
        topBar.addView(screenBtn);
        topBar.addView(buttonsBtn);
        topBar.addView(resetScreenBtn);
        topBar.addView(resetButtonsBtn);
        topBar.addView(doneBtn);
        root.addView(topBar, topBarLp);

        final boolean[] screenMode = {true};
        Runnable applyMode = () -> {
            screen.active = screenMode[0];
            touch.setTouchEnabled(!screenMode[0]);
            resetScreenBtn.setVisibility(screenMode[0] ? View.VISIBLE : View.GONE);
            resetButtonsBtn.setVisibility(screenMode[0] ? View.GONE : View.VISIBLE);
            screenBtn.setEnabled(!screenMode[0]);
            buttonsBtn.setEnabled(screenMode[0]);
        };
        screenBtn.setOnClickListener(v -> { screenMode[0] = true; applyMode.run(); });
        buttonsBtn.setOnClickListener(v -> { screenMode[0] = false; applyMode.run(); });
        resetScreenBtn.setOnClickListener(v -> screen.resetAndPersist());
        resetButtonsBtn.setOnClickListener(v -> touch.resetButtonLayout());
        doneBtn.setOnClickListener(v -> finish());
        applyMode.run();

        setContentView(root);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideSystemBars();
    }

    @SuppressWarnings("deprecation")
    private void hideSystemBars() {
        Window w = getWindow();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            w.setDecorFitsSystemWindows(false);
            WindowInsetsController c = w.getInsetsController();
            if (c != null) {
                c.hide(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
                c.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        } else {
            w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    // ---- settings.txt: same upsert-only read/rewrite pattern as OptionsActivity, duplicated rather than
    // shared since each copy is ~10 lines and the two screens' read/write needs don't otherwise overlap ----
    private static final Pattern KV = Pattern.compile("^\\s*([a-z_0-9]+)\\s*=\\s*(\\S+)\\s*$");

    private File settingsFile() {
        return AppFiles.resolve(this, "settings.txt");
    }

    private float readSetting(String key, float def) {
        File f = settingsFile();
        if (!f.exists()) return def;
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                Matcher m = KV.matcher(line);
                if (m.matches() && m.group(1).equals(key)) {
                    try {
                        return Float.parseFloat(m.group(2));
                    } catch (NumberFormatException ignored) {
                    }
                }
            }
        } catch (IOException ignored) {
        }
        return def;
    }

    void writeSetting(String key, String value) {
        File f = settingsFile();
        List<String> lines = new ArrayList<>();
        if (f.exists()) {
            try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = r.readLine()) != null) lines.add(line);
            } catch (IOException ignored) {
            }
        }
        Pattern p = Pattern.compile("^\\s*" + key + "\\s*=.*$");
        boolean found = false;
        for (int i = 0; i < lines.size(); i++) {
            if (p.matcher(lines.get(i)).matches()) {
                lines.set(i, key + " = " + value);
                found = true;
                break;
            }
        }
        if (!found) lines.add(key + " = " + value);
        File parent = f.getParentFile();
        if (parent != null) parent.mkdirs();
        try (FileWriter w = new FileWriter(f)) {
            for (String line : lines) w.write(line + "\n");
        } catch (IOException ignored) {
        }
    }

    void writeSetting(String key, float value) {
        writeSetting(key, String.format(Locale.US, "%.3f", value));
    }

    /**
     * The game's frame drawn at its real on-screen size/position (screen_fit=manual math, mirroring
     * PresentSlot() in vk_backend.cpp), directly draggable (pan) and pinchable (zoom). Visual feedback is
     * immediate every frame; settings.txt is only rewritten when a gesture ends (not on every touch-move),
     * since that's a full file read+rewrite and a drag can fire movement events far faster than disk I/O
     * should be asked to keep up with.
     */
    private class ScreenAdjustView extends View {
        boolean active = true;
        private float mZoomX, mZoomY, mOffX, mOffY;
        private float mLastX, mLastY;
        private boolean mDragging = false;
        private final ScaleGestureDetector mScaleDetector;
        private final Paint mFramePaint = new Paint();
        private final Paint mBorderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint mScreenBorderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint mHandlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

        // Current frame rect and its pre-zoom (aspect-fit) base size, recomputed every onDraw -- read by
        // onTouchEvent to hit-test the edge handles and to solve for zoom/offset while one is dragged.
        private float mBaseW, mBaseH, mFx, mFy, mFw, mFh;

        // Edge-handle drag: resizes one axis only, with the OPPOSITE edge held fixed (like a crop tool).
        // Two-finger pinch still scales both axes together in one gesture; this is how to do just one.
        private static final int DRAG_NONE = 0, DRAG_PAN = 1, DRAG_LEFT = 2, DRAG_RIGHT = 3, DRAG_TOP = 4,
                DRAG_BOTTOM = 5;
        private int mDragMode = DRAG_NONE;
        private float mAnchorX, mAnchorY;  // the edge opposite the one being dragged, fixed for the gesture
        private final float mHandleSlop;

        ScreenAdjustView(DisplayCalibrationActivity activity) {
            super(activity);
            mZoomX = clamp(readSetting("screen_zoom_x", 1.0f));
            mZoomY = clamp(readSetting("screen_zoom_y", 1.0f));
            mOffX = readSetting("screen_offset_x", 0.0f);
            mOffY = readSetting("screen_offset_y", 0.0f);
            mHandleSlop = dp(28);
            mFramePaint.setColor(0x802E7CD6);
            mBorderPaint.setStyle(Paint.Style.STROKE);
            mBorderPaint.setColor(0xFFFFFFFF);
            mBorderPaint.setStrokeWidth(3f);
            mScreenBorderPaint.setStyle(Paint.Style.STROKE);
            mScreenBorderPaint.setColor(0xFF505050);
            mScreenBorderPaint.setStrokeWidth(2f);
            mHandlePaint.setColor(0xFFFFD040);
            mScaleDetector = new ScaleGestureDetector(activity, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                @Override
                public boolean onScale(ScaleGestureDetector d) {
                    final float f = d.getScaleFactor();
                    mZoomX = clamp(mZoomX * f);
                    mZoomY = clamp(mZoomY * f);
                    invalidate();
                    return true;
                }

                @Override
                public void onScaleEnd(ScaleGestureDetector d) {
                    persist();
                }
            });
        }

        void resetAndPersist() {
            mZoomX = mZoomY = 1.0f;
            mOffX = mOffY = 0.0f;
            invalidate();
            persist();
        }

        private void persist() {
            writeSetting("screen_zoom_x", mZoomX);
            writeSetting("screen_zoom_y", mZoomY);
            writeSetting("screen_offset_x", mOffX);
            writeSetting("screen_offset_y", mOffY);
        }

        private float clamp(float z) {
            return Math.max(0.25f, Math.min(4.0f, z));
        }

        private float clampOffset(float o) {
            return Math.max(-1.0f, Math.min(1.0f, o));
        }

        /** Which edge handle (if any) is under (x, y); DRAG_PAN if none, for a plain move-the-frame drag. */
        private int hitHandle(float x, float y) {
            final float midY = mFy + mFh / 2f, midX = mFx + mFw / 2f;
            if (Math.abs(x - mFx) < mHandleSlop && Math.abs(y - midY) < mHandleSlop * 2) return DRAG_LEFT;
            if (Math.abs(x - (mFx + mFw)) < mHandleSlop && Math.abs(y - midY) < mHandleSlop * 2) return DRAG_RIGHT;
            if (Math.abs(y - mFy) < mHandleSlop && Math.abs(x - midX) < mHandleSlop * 2) return DRAG_TOP;
            if (Math.abs(y - (mFy + mFh)) < mHandleSlop && Math.abs(x - midX) < mHandleSlop * 2) return DRAG_BOTTOM;
            return DRAG_PAN;
        }

        @Override
        public boolean onTouchEvent(MotionEvent e) {
            if (!active) return false;
            mScaleDetector.onTouchEvent(e);
            switch (e.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    // A fresh single-finger touch grabs an edge handle if it lands on one, else pans the
                    // whole frame. A second finger landing (ACTION_POINTER_DOWN) always means a pinch, so
                    // it deliberately does not redo this hit-test.
                    mDragMode = hitHandle(e.getX(), e.getY());
                    if (mDragMode == DRAG_LEFT || mDragMode == DRAG_RIGHT)
                        mAnchorX = mDragMode == DRAG_LEFT ? mFx + mFw : mFx;
                    if (mDragMode == DRAG_TOP || mDragMode == DRAG_BOTTOM)
                        mAnchorY = mDragMode == DRAG_TOP ? mFy + mFh : mFy;
                    mLastX = e.getX();
                    mLastY = e.getY();
                    mDragging = true;
                    break;
                case MotionEvent.ACTION_POINTER_DOWN:
                    mLastX = e.getX();
                    mLastY = e.getY();
                    mDragging = true;
                    break;
                case MotionEvent.ACTION_MOVE: {
                    if (!mDragging || mScaleDetector.isInProgress() || e.getPointerCount() != 1) break;
                    final int sw = Math.max(1, getWidth()), sh = Math.max(1, getHeight());
                    switch (mDragMode) {
                        case DRAG_LEFT:
                        case DRAG_RIGHT: {
                            float newFw = mDragMode == DRAG_LEFT ? mAnchorX - e.getX() : e.getX() - mAnchorX;
                            newFw = Math.max(dp(40), newFw);
                            mZoomX = clamp(newFw / Math.max(1f, mBaseW));
                            final float clampedFw = mBaseW * mZoomX;
                            final float newFx = mDragMode == DRAG_LEFT ? mAnchorX - clampedFw : mAnchorX;
                            mOffX = clampOffset((newFx - (sw - clampedFw) / 2f) * 2f / sw);
                            invalidate();
                            break;
                        }
                        case DRAG_TOP:
                        case DRAG_BOTTOM: {
                            float newFh = mDragMode == DRAG_TOP ? mAnchorY - e.getY() : e.getY() - mAnchorY;
                            newFh = Math.max(dp(40), newFh);
                            mZoomY = clamp(newFh / Math.max(1f, mBaseH));
                            final float clampedFh = mBaseH * mZoomY;
                            final float newFy = mDragMode == DRAG_TOP ? mAnchorY - clampedFh : mAnchorY;
                            mOffY = clampOffset((newFy - (sh - clampedFh) / 2f) * 2f / sh);
                            invalidate();
                            break;
                        }
                        default: {  // DRAG_PAN
                            final float dx = e.getX() - mLastX, dy = e.getY() - mLastY;
                            mOffX = clampOffset(mOffX + dx / Math.max(1, sw / 2f));
                            mOffY = clampOffset(mOffY + dy / Math.max(1, sh / 2f));
                            invalidate();
                            break;
                        }
                    }
                    mLastX = e.getX();
                    mLastY = e.getY();
                    break;
                }
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_POINTER_UP:
                    persist();
                    // fall through
                case MotionEvent.ACTION_CANCEL:
                    mDragging = false;
                    mDragMode = DRAG_NONE;
                    break;
                default:
                    break;
            }
            return true;
        }

        @Override
        protected void onDraw(Canvas c) {
            final int sw = getWidth(), sh = getHeight();
            if (sw == 0 || sh == 0) return;
            final int pjw = 1280, pjh = 720;  // the game's own handheld frame size, same as PresentSlot()
            int bw = sw, bh = (int) ((long) sw * pjh / pjw);
            if (bh > sh) {
                bh = sh;
                bw = (int) ((long) sh * pjw / pjh);
            }
            mBaseW = bw;
            mBaseH = bh;
            mFw = bw * mZoomX;
            mFh = bh * mZoomY;
            mFx = (sw - mFw) / 2f + mOffX * sw / 2f;
            mFy = (sh - mFh) / 2f + mOffY * sh / 2f;
            c.drawRect(mFx, mFy, mFx + mFw, mFy + mFh, mFramePaint);
            c.drawRect(mFx, mFy, mFx + mFw, mFy + mFh, mBorderPaint);
            c.drawRect(1, 1, sw - 1, sh - 1, mScreenBorderPaint);
            if (active) {
                final float hr = dp(6);
                c.drawCircle(mFx, mFy + mFh / 2f, hr, mHandlePaint);
                c.drawCircle(mFx + mFw, mFy + mFh / 2f, hr, mHandlePaint);
                c.drawCircle(mFx + mFw / 2f, mFy, hr, mHandlePaint);
                c.drawCircle(mFx + mFw / 2f, mFy + mFh, hr, mHandlePaint);
            }
        }
    }
}
