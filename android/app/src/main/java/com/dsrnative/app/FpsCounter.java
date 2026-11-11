package com.dsrnative.app;

import android.app.Activity;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import java.util.Locale;

/**
 * The FPS counter (settings menu, fps_counter): a small label at the top centre of the screen, in its own
 * transparent window that Android's compositor puts over the game. It counts the frames the game presents (native
 * code) and updates twice a second, so the game's rendering does no extra work for it.
 */
public final class FpsCounter {
    private FpsCounter() {}

    private static final Handler sHandler = new Handler(Looper.getMainLooper());
    private static TextView sView;
    private static long sLastFrames, sLastNs;
    private static boolean sWanted;

    private static final Runnable sTick = new Runnable() {
        @Override
        public void run() {
            if (sView == null) return;
            long frames = 0;
            try {
                frames = SettingsMenu.nativeFrames();
            } catch (Throwable ignored) {
            }
            long now = SystemClock.elapsedRealtimeNanos();
            if (sLastNs != 0 && now > sLastNs) {
                double fps = (frames - sLastFrames) * 1e9 / (now - sLastNs);
                sView.setText(String.format(Locale.ROOT, "%.0f FPS", fps));
            }
            sLastFrames = frames;
            sLastNs = now;
            sHandler.postDelayed(this, 500);
        }
    };

    /** Shows (on = true) or removes the counter. */
    public static void apply(final Activity activity, final boolean on) {
        activity.runOnUiThread(() -> {
            sWanted = on;
            if (on)
                add(activity, 0);
            else
                remove(activity);
        });
    }

    private static void add(final Activity activity, final int attempt) {
        if (sView != null || !sWanted) return;
        final View decor = activity.getWindow().getDecorView();
        if (decor.getWindowToken() == null) {  // the game's window is not attached yet
            if (attempt < 50) sHandler.postDelayed(() -> add(activity, attempt + 1), 100);
            return;
        }
        final float dp = activity.getResources().getDisplayMetrics().density;
        TextView t = new TextView(activity);
        t.setText("— FPS");
        t.setTextSize(12);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        t.setTextColor(0xFFF2E3B6);
        t.setGravity(Gravity.CENTER);
        t.setPadding(Math.round(8 * dp), Math.round(2 * dp), Math.round(8 * dp), Math.round(2 * dp));
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(0x99000000);
        bg.setCornerRadius(6 * dp);
        t.setBackground(bg);
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        lp.token = decor.getWindowToken();
        // Top centre, between the on-screen - and + buttons (and clear of the game's HUD).
        lp.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
        lp.y = Math.round(4 * dp);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        try {
            activity.getWindowManager().addView(t, lp);
            sView = t;
            sLastNs = 0;
            sHandler.removeCallbacks(sTick);
            sHandler.post(sTick);
        } catch (Exception e) {
            android.util.Log.w("DSRNative", "fps counter: could not add the overlay", e);
        }
    }

    private static void remove(Activity activity) {
        sHandler.removeCallbacks(sTick);
        if (sView == null) return;
        try {
            activity.getWindowManager().removeView(sView);
        } catch (Exception e) {
            android.util.Log.w("DSRNative", "fps counter: could not remove the overlay", e);
        }
        sView = null;
    }
}
