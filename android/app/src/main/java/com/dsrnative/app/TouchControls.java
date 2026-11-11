package com.dsrnative.app;

import android.content.Context;
import android.graphics.Color;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.github.controlwear.virtual.joystick.android.JoystickView;

/**
 * On-screen touch controls: two library-provided analog sticks (movement, camera) plus plain Android
 * buttons for everything else. Default positions are fixed gravity/margins; long-press the toggle
 * button (top-right) to enter Edit Mode, drag anything to reposition it, long-press again to save and
 * resume play. Positions persist to touch_layout.txt next to settings.txt.
 *
 * See GameActivity.addTouchOverlay() for why this has to be added as a separate window rather than
 * drawn into the game's own surface.
 */
public class TouchControls extends FrameLayout {
    // NativeActivity loads libdsr.so itself via a raw dlopen in its own native bootstrap, which never
    // registers the library with this classloader's JNI native-method list -- so a *different* Java
    // class trying to call a native method here finds nothing, even though the library is already
    // mapped in the process. Loading it again is cheap (same cached handle) and fixes that lookup.
    static {
        System.loadLibrary("dsr");
    }

    // nn::hid::NpadButton bit positions (must match dsr::input::Button in src/hle/input.h).
    private static final long BTN_A = 1L, BTN_B = 1L << 1, BTN_X = 1L << 2, BTN_Y = 1L << 3;
    private static final long BTN_STICK_L = 1L << 4, BTN_STICK_R = 1L << 5;
    private static final long BTN_L = 1L << 6, BTN_R = 1L << 7, BTN_ZL = 1L << 8, BTN_ZR = 1L << 9;
    private static final long BTN_PLUS = 1L << 10, BTN_MINUS = 1L << 11;
    private static final long BTN_LEFT = 1L << 12, BTN_UP = 1L << 13, BTN_RIGHT = 1L << 14, BTN_DOWN = 1L << 15;

    private static final String TAG = "TouchControls";

    private static native void nativeSetTouchButtons(long mask);
    private static native void nativeSetTouchStick(int which, int x, int y);

    private long mButtonMask = 0;
    private View mButtonsLayer, mSticksLayer;
    private boolean mVisible = true;
    private boolean mEditMode = false;

    // Every repositionable control (both sticks, all buttons -- not the toggle, which stays put), in
    // the coordinate space of this view: each sits in a full-bleed child layer (mSticksLayer /
    // mButtonsLayer), which is why a child's getLeft()/getTop() already equal its position here too.
    private final Map<String, View> mControls = new LinkedHashMap<>();

    // Edit-mode drag tracking: pointer id -> the control it grabbed, and the finger's offset from
    // that control's top-left corner (so dragging doesn't snap the control to the touch point).
    private String mDragId;
    private int mDragOffsetX, mDragOffsetY;
    private View mToggle;

    public TouchControls(Context c) {
        super(c);
        // Same Mali/MediaTek gralloc issue as before: hardware-accelerated compositing of a view
        // stacked as a separate window over a NativeActivity's Vulkan surface fails silently on this
        // GPU. Software compositing avoids that path.
        setLayerType(LAYER_TYPE_SOFTWARE, null);

        mSticksLayer = buildSticks();
        mButtonsLayer = buildButtons();
        addView(mSticksLayer);
        addView(mButtonsLayer);
        addToggle();

        loadLayout();
    }

    private View buildSticks() {
        FrameLayout layer = new FrameLayout(getContext());
        final int size = dp(120);

        JoystickView move = new JoystickView(getContext());
        styleStick(move);
        FrameLayout.LayoutParams mlp = new FrameLayout.LayoutParams(size, size);
        mlp.gravity = Gravity.BOTTOM | Gravity.START;
        mlp.setMargins(dp(24), 0, 0, dp(24));
        layer.addView(move, mlp);
        mControls.put("LSTICK", move);
        // angle: 0-360 counter-clockwise from the +x axis; strength: 0-100% of the way to the border.
        // -32767..32767, +y = up (matches dsr::input::PadState / AxisToStick in android_main.cpp).
        move.setOnMoveListener((angle, strength) -> {
            Log.d(TAG, "left stick onMove angle=" + angle + " strength=" + strength + " buttonMask=0x"
                    + Long.toHexString(mButtonMask));
            final double rad = Math.toRadians(angle);
            final float mag = strength / 100f * 32767f;
            nativeSetTouchStick(0, (int) (Math.cos(rad) * mag), (int) (Math.sin(rad) * mag));
        }, 16);

        JoystickView look = new JoystickView(getContext());
        styleStick(look);
        FrameLayout.LayoutParams llp = new FrameLayout.LayoutParams(size, size);
        llp.gravity = Gravity.BOTTOM | Gravity.END;
        llp.setMargins(0, 0, dp(24), dp(24));
        layer.addView(look, llp);
        mControls.put("RSTICK", look);
        look.setOnMoveListener((angle, strength) -> {
            final double rad = Math.toRadians(angle);
            final float mag = strength / 100f * 32767f;
            nativeSetTouchStick(1, (int) (Math.cos(rad) * mag), (int) (Math.sin(rad) * mag));
        }, 16);

        // L3/R3: dedicated small buttons next to each stick, not "press the stick itself" -- that
        // would fire on every touch-down used to start moving the stick, not just a deliberate click.
        addButton(layer, "STICK_L", "L3", BTN_STICK_L, Gravity.BOTTOM | Gravity.START, dp(24) + size + dp(8), 0, 0,
                dp(24) + size - dp(40), dp(40));
        addButton(layer, "STICK_R", "R3", BTN_STICK_R, Gravity.BOTTOM | Gravity.END, 0, 0, dp(24) + size + dp(8),
                dp(24) + size - dp(40), dp(40));

        return layer;
    }

    // The library defaults background/border to fully transparent and only the small drag-knob to
    // black -- with no styling at all, that's an all-but-invisible dot on a dark game scene. These
    // match the translucent-white look of the plain buttons elsewhere in this overlay.
    private void styleStick(JoystickView stick) {
        stick.setBackgroundColor(Color.argb(60, 255, 255, 255));
        stick.setBorderColor(Color.argb(160, 255, 255, 255));
        stick.setBorderWidth(dp(2));
        stick.setButtonColor(Color.argb(220, 255, 255, 255));
    }

    private View buildButtons() {
        FrameLayout layer = new FrameLayout(getContext());
        final int btn = dp(52), gap = dp(8);

        // D-pad, above the left stick.
        addButton(layer, "DPAD_L", "◀", BTN_LEFT, Gravity.BOTTOM | Gravity.START, dp(24) - btn - gap, 0, 0, dp(168), btn);
        addButton(layer, "DPAD_U", "▲", BTN_UP, Gravity.BOTTOM | Gravity.START, dp(24), 0, 0, dp(168) + btn + gap, btn);
        addButton(layer, "DPAD_R", "▶", BTN_RIGHT, Gravity.BOTTOM | Gravity.START, dp(24) + btn + gap, 0, 0, dp(168), btn);
        addButton(layer, "DPAD_D", "▼", BTN_DOWN, Gravity.BOTTOM | Gravity.START, dp(24), 0, 0, dp(168) - btn - gap, btn);

        // Face buttons, above the right stick (Switch layout: A right, B bottom, X top, Y left).
        addButton(layer, "FACE_A", "A", BTN_A, Gravity.BOTTOM | Gravity.END, 0, 0, dp(24) - btn - gap, dp(168), btn);
        addButton(layer, "FACE_B", "B", BTN_B, Gravity.BOTTOM | Gravity.END, 0, 0, dp(24), dp(168) - btn - gap, btn);
        addButton(layer, "FACE_X", "X", BTN_X, Gravity.BOTTOM | Gravity.END, 0, 0, dp(24), dp(168) + btn + gap, btn);
        addButton(layer, "FACE_Y", "Y", BTN_Y, Gravity.BOTTOM | Gravity.END, 0, 0, dp(24) + btn + gap, dp(168), btn);

        // Shoulders, top corners; +/- top centre.
        addButton(layer, "BTN_ZL", "ZL", BTN_ZL, Gravity.TOP | Gravity.START, dp(12), dp(12), 0, 0, btn);
        addButton(layer, "BTN_L", "L", BTN_L, Gravity.TOP | Gravity.START, dp(12) + btn + gap, dp(12), 0, 0, btn);
        addButton(layer, "BTN_ZR", "ZR", BTN_ZR, Gravity.TOP | Gravity.END, 0, dp(12), dp(12), 0, btn);
        addButton(layer, "BTN_R", "R", BTN_R, Gravity.TOP | Gravity.END, 0, dp(12), dp(12) + btn + gap, 0, btn);
        addButton(layer, "BTN_MINUS", "-", BTN_MINUS, Gravity.TOP | Gravity.CENTER_HORIZONTAL, -dp(44), dp(12), 0, 0, dp(44));
        addButton(layer, "BTN_PLUS", "+", BTN_PLUS, Gravity.TOP | Gravity.CENTER_HORIZONTAL, dp(44), dp(12), 0, 0, dp(44));

        return layer;
    }

    private void addButton(FrameLayout layer, String id, String label, long bit, int gravity, int left, int top,
                            int right, int bottom, int size) {
        Button b = new Button(getContext());
        b.setText(label);
        b.setAllCaps(false);
        b.setBackgroundColor(Color.argb(96, 255, 255, 255));
        b.setTextColor(Color.argb(220, 255, 255, 255));
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(size, size);
        lp.gravity = gravity;
        lp.setMargins(left, top, right, bottom);
        b.setOnTouchListener((v, event) -> {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    mButtonMask |= bit;
                    nativeSetTouchButtons(mButtonMask);
                    Log.d(TAG, "button " + id + " DOWN, pointers now active=" + event.getPointerCount()
                            + " mask=0x" + Long.toHexString(mButtonMask));
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    mButtonMask &= ~bit;
                    nativeSetTouchButtons(mButtonMask);
                    Log.d(TAG, "button " + id + " UP/CANCEL mask=0x" + Long.toHexString(mButtonMask));
                    return true;
                default:
                    return false;
            }
        });
        layer.addView(b, lp);
        mControls.put(id, b);
    }

    private void addToggle() {
        Button toggle = new Button(getContext());
        toggle.setText("⊡");
        toggle.setBackgroundColor(Color.argb(140, 64, 64, 64));
        toggle.setTextColor(Color.WHITE);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(dp(40), dp(40));
        lp.gravity = Gravity.TOP | Gravity.END;
        lp.setMargins(0, dp(60), dp(8), 0);  // clear of the ZR/R buttons above
        toggle.setOnClickListener(v -> {
            if (mEditMode) return;  // tap does nothing while editing; use long-press to exit
            mVisible = !mVisible;
            mSticksLayer.setVisibility(mVisible ? VISIBLE : GONE);
            mButtonsLayer.setVisibility(mVisible ? VISIBLE : GONE);
            if (!mVisible) releaseAllInputs();
        });
        toggle.setOnLongClickListener(v -> {
            mEditMode = !mEditMode;
            Log.i(TAG, "toggle long-pressed, edit mode now " + mEditMode);
            toggle.setBackgroundColor(mEditMode ? Color.argb(180, 255, 180, 0) : Color.argb(140, 64, 64, 64));
            releaseAllInputs();
            if (mEditMode) {
                mVisible = true;
                mSticksLayer.setVisibility(VISIBLE);
                mButtonsLayer.setVisibility(VISIBLE);
                Toast.makeText(getContext(), "Edit mode: drag to move, long-press ⊡ again to save",
                        Toast.LENGTH_SHORT).show();
            } else {
                saveLayout();
            }
            return true;
        });
        addView(toggle, lp);
        mToggle = toggle;
    }

    private void releaseAllInputs() {
        mButtonMask = 0;
        nativeSetTouchButtons(0);
        nativeSetTouchStick(0, 0, 0);
        nativeSetTouchStick(1, 0, 0);
    }

    // ---- edit mode: intercept everything so children never see the drag (no accidental presses /
    // joystick knob drags while repositioning), reposition by writing a fresh LayoutParams directly. --
    @Override
    public boolean onInterceptTouchEvent(MotionEvent event) {
        if (!mEditMode) return super.onInterceptTouchEvent(event);
        // Never intercept over the toggle itself -- otherwise, once edit mode starts, the long-press
        // that's supposed to exit it can never reach the toggle's own listener again.
        final int x = (int) event.getX(), y = (int) event.getY();
        if (mToggle != null && x >= mToggle.getLeft() && x <= mToggle.getRight() && y >= mToggle.getTop() &&
            y <= mToggle.getBottom())
            return false;
        return true;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!mEditMode) return super.onTouchEvent(event);
        final int x = (int) event.getX(), y = (int) event.getY();
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                mDragId = findControlAt(x, y);
                if (mDragId != null) {
                    View v = mControls.get(mDragId);
                    mDragOffsetX = x - v.getLeft();
                    mDragOffsetY = y - v.getTop();
                }
                return mDragId != null;
            case MotionEvent.ACTION_MOVE:
                if (mDragId == null) return false;
                moveControl(mDragId, x - mDragOffsetX, y - mDragOffsetY);
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                mDragId = null;
                return true;
            default:
                return true;
        }
    }

    private String findControlAt(int x, int y) {
        for (Map.Entry<String, View> e : mControls.entrySet()) {
            View v = e.getValue();
            if (x >= v.getLeft() && x <= v.getRight() && y >= v.getTop() && y <= v.getBottom()) return e.getKey();
        }
        return null;
    }

    private void moveControl(String id, int left, int top) {
        View v = mControls.get(id);
        if (v == null) return;
        left = clamp(left, 0, getWidth() - v.getWidth());
        top = clamp(top, 0, getHeight() - v.getHeight());
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(v.getWidth(), v.getHeight());
        lp.gravity = Gravity.TOP | Gravity.START;
        lp.setMargins(left, top, 0, 0);
        v.setLayoutParams(lp);
    }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    // ---- persistence: touch_layout.txt, "pos_<id> = fx fy" (top-left corner, fraction of this
    // view's own width/height) for whichever controls were ever dragged -- untouched ones just keep
    // their default gravity/margin layout from build{Sticks,Buttons}() above. ------------------------
    private static final Pattern KV = Pattern.compile("^\\s*pos_([A-Z_]+)\\s*=\\s*([0-9.]+)\\s+([0-9.]+)\\s*$");

    private void loadLayout() {
        File f = AppFiles.resolve(getContext(), "touch_layout.txt");
        Log.i(TAG, "loading layout from " + f.getAbsolutePath() + " exists=" + f.exists());
        if (!f.exists()) return;
        // Positions are fractions of this view's size, which isn't known until after layout.
        post(() -> {
            try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = r.readLine()) != null) {
                    Matcher m = KV.matcher(line);
                    if (!m.matches()) continue;
                    View v = mControls.get(m.group(1));
                    if (v == null) continue;
                    final float fx = parseF(m.group(2)), fy = parseF(m.group(3));
                    moveControl(m.group(1), (int) (fx * getWidth()), (int) (fy * getHeight()));
                }
            } catch (IOException ignored) {
            }
        });
    }

    private void saveLayout() {
        File f = AppFiles.resolve(getContext(), "touch_layout.txt");
        Log.i(TAG, "saving layout to " + f.getAbsolutePath());
        File parent = f.getParentFile();
        if (parent != null) {
            final boolean madeDirs = parent.mkdirs();
            Log.i(TAG, "parent dir " + parent.getAbsolutePath() + " exists=" + parent.exists()
                    + " (mkdirs created new=" + madeDirs + ")");
        }
        try (FileWriter w = new FileWriter(f)) {
            w.write("# On-screen control positions (long-press the toggle button, top-right, to edit in-game)\n");
            for (Map.Entry<String, View> e : mControls.entrySet()) {
                View v = e.getValue();
                w.write("pos_" + e.getKey() + " = " + (v.getLeft() / (float) getWidth()) + " "
                        + (v.getTop() / (float) getHeight()) + "\n");
            }
            Log.i(TAG, "layout saved: " + mControls.size() + " controls, file now " + f.length() + " bytes");
            Toast.makeText(getContext(), "Layout saved", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            // Not fatal to play -- the layout just reverts to defaults (or whatever was last saved)
            // next launch -- but silently swallowing this is exactly how "it doesn't save" becomes
            // impossible to diagnose, so it's logged and surfaced instead.
            Log.e(TAG, "failed to save layout to " + f.getAbsolutePath(), e);
            Toast.makeText(getContext(), "Couldn't save layout: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private static float parseF(String s) {
        try { return Float.parseFloat(s); } catch (NumberFormatException e) { return 0f; }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}
