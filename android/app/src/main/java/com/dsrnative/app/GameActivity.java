package com.dsrnative.app;

import android.Manifest;
import android.app.NativeActivity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.TextView;
import java.util.Locale;

/**
 * The native activity, full screen: the status and navigation bars are hidden while the game runs (immersive
 * mode: a swipe from the screen edge shows them for a moment), and the game may draw under a display cutout.
 */
public class GameActivity extends NativeActivity {
    // NativeActivity's own dlopen of this library (driven by the manifest's android:value) doesn't
    // register it with this classloader, so a plain `native` method here (nativeGetFps) wouldn't
    // resolve without this -- same fix as TouchControls needed for its JNI calls.
    static { System.loadLibrary("dsr"); }
    // Access to Download/DSRNative (game files, saves, settings): 0 = waiting for the player's answer,
    // 1 = granted, 2 = refused (the game then falls back to the app's private folder). Read by native code.
    private static volatile int sStorage = 0;
    private boolean mAsked = false, mPausedSinceAsk = false;

    public static int storageAccess() {
        return sStorage;
    }

    private boolean hasStorageAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) return Environment.isExternalStorageManager();
        return checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestStorageAccess() {
        if (hasStorageAccess()) {
            sStorage = 1;
            return;
        }
        mAsked = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            } catch (Exception e) {
                try {
                    startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
                } catch (Exception e2) {
                    sStorage = 2;
                }
            }
        } else {
            requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        sStorage = hasStorageAccess() ? 1 : 2;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mAsked) mPausedSinceAsk = true;
        mFpsHandler.removeCallbacks(mFpsTick);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestStorageAccess();
        Window w = getWindow();
        w.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            WindowManager.LayoutParams lp = w.getAttributes();
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            w.setAttributes(lp);
        }
        hideSystemBars();
    }

    private TouchControls mOverlay;
    private TextView mFpsView;
    private final Handler mFpsHandler = new Handler(Looper.getMainLooper());
    private native float nativeGetFps();
    private final Runnable mFpsTick = new Runnable() {
        @Override
        public void run() {
            if (mFpsView != null) mFpsView.setText(String.format(Locale.US, "%.0f fps", nativeGetFps()));
            mFpsHandler.postDelayed(this, 500);
        }
    };

    // On-screen FPS counter: a second small panel, same approach as the touch overlay (its own window,
    // not drawn into the activity's content view -- see addTouchOverlay()'s comment for why).
    private void addFpsOverlay() {
        if (!fpsCounterEnabled()) return;
        if (mFpsView == null) {
            mFpsView = new TextView(this);
            mFpsView.setTextColor(Color.GREEN);
            mFpsView.setTextSize(14);
            mFpsView.setPadding(dp(6), dp(2), dp(6), dp(2));
            mFpsView.setBackgroundColor(0x80000000);
            WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_PANEL,
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                    PixelFormat.TRANSLUCENT);
            lp.gravity = Gravity.TOP | Gravity.START;
            lp.token = getWindow().getDecorView().getWindowToken();
            getWindowManager().addView(mFpsView, lp);
        }
        mFpsHandler.removeCallbacks(mFpsTick);
        mFpsHandler.post(mFpsTick);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    // settings.txt's fps_counter key (set by OptionsActivity); defaults to on if missing.
    private boolean fpsCounterEnabled() {
        java.io.File f = AppFiles.resolve(this, "settings.txt");
        if (!f.exists()) return true;
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("fps_counter") && line.contains("="))
                    return !line.substring(line.indexOf('=') + 1).trim().equals("0");
            }
        } catch (java.io.IOException ignored) {
        }
        return true;
    }

    // On-screen touch controls: a genuinely separate window (TYPE_APPLICATION_PANEL), composited as
    // its own layer above the activity's main surface. Drawing into the activity's own content view
    // was tried first and failed: that view shares the same surface the Vulkan swapchain presents to
    // directly, so the two were racing for one buffer instead of being layered, and the overlay's
    // content got overwritten before it was ever visible. A sub-window needs the activity's window
    // token, which isn't valid until the window actually has focus, so this can't happen in onCreate.
    private void addTouchOverlay() {
        if (mOverlay != null || !touchControlsEnabled()) return;
        mOverlay = new TouchControls(this);
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_PANEL, 0, PixelFormat.TRANSLUCENT);
        lp.token = getWindow().getDecorView().getWindowToken();
        getWindowManager().addView(mOverlay, lp);
    }

    // settings.txt's touch_controls key (set by OptionsActivity); defaults to on if missing.
    private boolean touchControlsEnabled() {
        java.io.File f = AppFiles.resolve(this, "settings.txt");
        if (!f.exists()) return true;
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("touch_controls") && line.contains("="))
                    return !line.substring(line.indexOf('=') + 1).trim().equals("0");
            }
        } catch (java.io.IOException ignored) {
        }
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Back from the "All files access" settings screen: the answer is final for this launch.
        if (mAsked && mPausedSinceAsk && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
            sStorage = hasStorageAccess() ? 1 : 2;
        hideSystemBars();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            hideSystemBars();
            addTouchOverlay();
            addFpsOverlay();
        }
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
}
