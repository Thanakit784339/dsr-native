package com.dsrnative.app;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.NativeActivity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;

/**
 * The native activity, full screen: the status and navigation bars are hidden while the game runs (immersive
 * mode: a swipe from the screen edge shows them for a moment), and the game may draw under a display cutout.
 */
public class GameActivity extends NativeActivity {
    // Access to Download/DSRNative (game files, saves, settings): 0 = waiting for the player's answer,
    // 1 = granted, 2 = refused (the game then falls back to the app's private folder). Read by native code.
    private static volatile int sStorage = 0;
    private boolean mAsked = false, mPausedSinceAsk = false;

    public static int storageAccess() {
        return sStorage;
    }

    // ---- game folder chosen by the player (when the game is not in Download/DSRNative/game) -----------------
    static final int REQ_GAME_FOLDER = 4103;
    // 0 = not asked, 1 = the player is choosing (dialog or picker up), 2 = chosen (takeGameFolder). Read by native.
    private static volatile int sFolderState = 0;
    private static volatile String sFolder = null;

    public static int gameFolderState() {
        return sFolderState;
    }

    /** The folder the player chose (holding exefs/main), once; null if none yet. */
    public static String takeGameFolder() {
        if (sFolderState != 2) return null;
        String f = sFolder;
        sFolder = null;
        sFolderState = 0;
        return f;
    }

    /** Called by native code when the game files are not found, or found incomplete. */
    public void askGameFolder(final String title, final String message) {
        sFolderState = 1;
        runOnUiThread(() -> showFolderDialog(title, message));
    }

    private void showFolderDialog(String title, String message) {
        AlertDialog d = new AlertDialog.Builder(this, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setTitle(title)
                .setMessage(message)
                .setCancelable(false)
                .setPositiveButton("Choose folder", (dlg, which) -> pickGameFolder())
                .setNegativeButton("Exit", (dlg, which) -> finishAndRemoveTask())
                .create();
        showImmersive(d);
    }

    private void pickGameFolder() {
        try {
            startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_GAME_FOLDER);
        } catch (Exception e) {
            showFolderDialog("Game files not found", "No file manager is available to choose a folder. Copy the "
                    + "complete game folder to Download/DSRNative/game instead, then start the app again.");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (SettingsMenu.onActivityResult(this, requestCode, resultCode, data)) return;
        if (requestCode != REQ_GAME_FOLDER) return;
        Uri uri = resultCode == RESULT_OK && data != null ? data.getData() : null;
        if (uri == null) {
            showFolderDialog("Game files not found", "No folder was chosen. Choose the folder that holds the game "
                    + "files: the one that contains the exefs folder, dvdbnd0.bdt and dvdbnd1.bdt.");
            return;
        }
        String path = SettingsMenu.treeToPath(uri);
        String root = path == null ? null : SettingsMenu.findGameRoot(path);
        if (root == null) {
            String partial = path == null ? null : SettingsMenu.findPartialGameRoot(path);
            if (partial != null)
                showFolderDialog("Game files incomplete", partial + " is missing:\n"
                        + SettingsMenu.missingGameFiles(partial) + "\n\nThese files belong next to the exefs folder. "
                        + "Copy the complete game folder (the base game with the 1.0.3 update merged in), or choose "
                        + "a folder that holds all of it.");
            else
                showFolderDialog("Game files not found", (path == null ? "That folder" : path)
                        + " does not hold the game files.\n\nChoose the folder that contains the exefs folder, "
                        + "dvdbnd0.bdt and dvdbnd1.bdt.");
            return;
        }
        sFolder = root;
        sFolderState = 2;  // native code picks it up when the activity resumes
    }

    // ---- dialogs over the full-screen game ------------------------------------------------------------------
    /** Shows a dialog without bringing back the status and navigation bars. */
    static void showImmersive(Dialog d) {
        Window w = d.getWindow();
        if (w == null) {
            d.show();
            return;
        }
        // Not focusable while it appears (the bars would show), then focusable for touch and controller input.
        w.addFlags(WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE);
        d.show();
        hideBars(w, false);
        w.clearFlags(WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE);
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
        if (hasFocus) hideSystemBars();
    }

    private void hideSystemBars() {
        hideBars(getWindow(), true);
    }

    @SuppressWarnings("deprecation")
    static void hideBars(Window w, boolean underBars) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (underBars) w.setDecorFitsSystemWindows(false);
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
