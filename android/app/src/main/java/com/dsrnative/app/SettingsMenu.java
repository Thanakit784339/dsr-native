package com.dsrnative.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.net.Uri;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.OpenableColumns;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The in-game settings menu, opened by holding - and + for a second (controller or on-screen buttons). Two tabs:
 * Driver (the GPU driver used from the next start) and Settings. Choices are saved to settings.txt at once; those
 * that can change while the game runs are applied at once (native code), the others at the next start (a message
 * says so). The game is held while the menu shows. Works with touch and with a controller (D-pad / stick to move,
 * A to choose, B to close, L / R to switch tabs).
 */
public final class SettingsMenu {
    /** Applies a setting now if it can be; false = at the next start. */
    static native boolean nativeApply(String key, String value);
    /** The menu opened (true) or closed: the game is held while it shows. */
    static native void nativeMenuOpen(boolean open);
    /** Frames shown on screen so far (the FPS counter). */
    static native long nativeFrames();

    static final int REQ_DRIVER = 4101, REQ_FOLDER = 4102;

    private static String sSettingsPath = "", sDriversDir = "", sRunningDriver = "system", sGameDir = "",
            sDefaultGameDir = "";
    private static SettingsMenu sOpen;

    /** Called by native code at game start. */
    public static void configure(Activity activity, String settingsPath, String driversDir, String runningDriver,
            String gameDir, String defaultGameDir) {
        sSettingsPath = settingsPath;
        sDriversDir = driversDir;
        sRunningDriver = runningDriver == null || runningDriver.isEmpty() ? "system" : runningDriver;
        sGameDir = gameDir;
        sDefaultGameDir = defaultGameDir;
        Map<String, String> v = readSettings();
        TouchControls.sRecenter = !"0".equals(v.get("touch_camera_recenter"));
        if ("1".equals(v.get("fps_counter"))) FpsCounter.apply(activity, true);
    }

    /** Called by native code when - and + have been held for a second. */
    public static void show(final Activity activity) {
        activity.runOnUiThread(() -> {
            if (sOpen != null) return;
            try {
                new SettingsMenu(activity).open();
            } catch (Exception e) {
                android.util.Log.w("DSRNative", "settings menu: could not open", e);
                sOpen = null;
                nativeMenuOpen(false);
            }
        });
    }

    /** From GameActivity.onActivityResult: true if the result was for the menu. */
    static boolean onActivityResult(Activity activity, int requestCode, int resultCode, Intent data) {
        if (requestCode != REQ_DRIVER && requestCode != REQ_FOLDER) return false;
        Uri uri = resultCode == Activity.RESULT_OK && data != null ? data.getData() : null;
        if (sOpen == null || uri == null) return true;
        if (requestCode == REQ_DRIVER)
            sOpen.importDriver(uri);
        else
            sOpen.chooseGameFolder(uri);
        return true;
    }

    // ---- settings.txt -------------------------------------------------------------------------------------------

    /** key -> value of settings.txt (the last line for a key wins, as in native code). */
    static Map<String, String> readSettings() {
        Map<String, String> m = new HashMap<>();
        try (BufferedReader r = new BufferedReader(new FileReader(sSettingsPath))) {
            String line;
            while ((line = r.readLine()) != null) {
                String t = line.trim();
                if (t.isEmpty() || t.startsWith("#")) continue;
                int eq = t.indexOf('=');
                if (eq <= 0) continue;
                m.put(t.substring(0, eq).trim(), t.substring(eq + 1).trim());
            }
        } catch (IOException e) {
            android.util.Log.w("DSRNative", "settings menu: cannot read " + sSettingsPath, e);
        }
        return m;
    }

    /** Sets key = value in settings.txt: every line of the key is replaced, or a line appended. */
    static boolean writeSetting(String key, String value) {
        List<String> lines = new ArrayList<>();
        File f = new File(sSettingsPath);
        if (f.exists()) {
            try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = r.readLine()) != null) lines.add(line);
            } catch (IOException e) {
                return false;
            }
        }
        boolean done = false;
        for (int i = 0; i < lines.size(); i++) {
            String t = lines.get(i).trim();
            if (t.startsWith("#")) continue;
            int eq = t.indexOf('=');
            if (eq <= 0 || !t.substring(0, eq).trim().equals(key)) continue;
            lines.set(i, key + " = " + value);
            done = true;
        }
        if (!done) {
            lines.add("");
            lines.add(key + " = " + value);
        }
        try (FileWriter w = new FileWriter(f, false)) {
            for (String l : lines) {
                w.write(l);
                w.write('\n');
            }
            return true;
        } catch (IOException e) {
            android.util.Log.w("DSRNative", "settings menu: cannot write " + sSettingsPath, e);
            return false;
        }
    }

    // ---- folders ------------------------------------------------------------------------------------------------

    /** A folder picked with ACTION_OPEN_DOCUMENT_TREE as a file path (the app has "All files access"). */
    static String treeToPath(Uri uri) {
        try {
            String id = DocumentsContract.getTreeDocumentId(uri);
            if (id == null) return null;
            if (id.startsWith("raw:")) return id.substring(4);
            if ("com.android.providers.downloads.documents".equals(uri.getAuthority()) && "downloads".equals(id))
                return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getPath();
            int c = id.indexOf(':');
            if (c < 0) return null;
            String volume = id.substring(0, c), rel = id.substring(c + 1);
            String root = "primary".equalsIgnoreCase(volume) ? Environment.getExternalStorageDirectory().getPath()
                                                             : "/storage/" + volume;
            while (rel.endsWith("/")) rel = rel.substring(0, rel.length() - 1);
            return rel.isEmpty() ? root : root + "/" + rel;
        } catch (Exception e) {
            return null;
        }
    }

    /** The game's root in or around a chosen folder (the folder holding exefs/main), or null. */
    static String findGameRoot(String path) {
        if (path == null) return null;
        File p = new File(path);
        if (new File(p, "exefs/main").isFile()) return p.getPath();
        File game = new File(p, "game");  // the folder above it, as Download/DSRNative
        if (new File(game, "exefs/main").isFile()) return game.getPath();
        File parent = p.getParentFile();  // the exefs folder itself
        if (p.getName().equals("exefs") && parent != null && new File(p, "main").isFile()) return parent.getPath();
        return null;
    }

    // ---- the menu -----------------------------------------------------------------------------------------------

    private static final int GOLD = 0xFFC9A55C, PANEL = 0xF2121212, CARD = 0xFF242424, TEXT = 0xFFEDEDED,
            DIM = 0xFF9A9A9A;

    private final Activity mA;
    private final float mDp;
    private Dialog mDialog;
    private LinearLayout mBody;
    private ScrollView mScroll;
    private TextView mTabDriver, mTabSettings;
    private int mTab = 0;
    private Map<String, String> mValues = new HashMap<>();

    private SettingsMenu(Activity a) {
        mA = a;
        mDp = a.getResources().getDisplayMetrics().density;
    }

    private int dp(float v) {
        return Math.round(v * mDp);
    }

    private void open() {
        sOpen = this;
        nativeMenuOpen(true);
        TouchControls.releaseForMenu();
        mValues = readSettings();

        LinearLayout root = new LinearLayout(mA);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(18), dp(14), dp(18), dp(12));
        root.setBackground(rounded(PANEL, 14, 0xFF4A3D22));

        LinearLayout header = new LinearLayout(mA);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = text("DSR Native", 20, GOLD, true);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        mTabDriver = chip("Driver", true);
        mTabSettings = chip("Settings", true);
        mTabDriver.setOnClickListener(v -> showTab(0));
        mTabSettings.setOnClickListener(v -> showTab(1));
        header.addView(mTabDriver);
        header.addView(mTabSettings);
        TextView close = chip("Close", true);
        close.setOnClickListener(v -> mDialog.dismiss());
        LinearLayout.LayoutParams clp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        clp.leftMargin = dp(18);
        header.addView(close, clp);
        root.addView(header);

        TextView hint = text("Hold − and + for a second to open this menu.  Controller: D-pad to move, A to choose, "
                + "B to close, L / R to switch tabs.", 12, DIM, false);
        hint.setPadding(0, dp(4), 0, dp(8));
        root.addView(hint);

        mScroll = new ScrollView(mA);
        mScroll.setFillViewport(true);
        mBody = new LinearLayout(mA);
        mBody.setOrientation(LinearLayout.VERTICAL);
        mScroll.addView(mBody, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(mScroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        mDialog = new Dialog(mA);
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        mDialog.setContentView(root);
        Window w = mDialog.getWindow();
        if (w != null) w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        mDialog.setCanceledOnTouchOutside(true);
        mDialog.setOnDismissListener(d -> {
            sOpen = null;
            nativeMenuOpen(false);
        });
        mDialog.setOnKeyListener((d, keyCode, e) -> {
            boolean down = e.getAction() == KeyEvent.ACTION_DOWN;
            switch (keyCode) {
                case KeyEvent.KEYCODE_BUTTON_B:
                case KeyEvent.KEYCODE_BACK:
                    if (down && e.getRepeatCount() == 0) mDialog.dismiss();
                    return true;
                case KeyEvent.KEYCODE_BUTTON_A: {
                    if (down && e.getRepeatCount() == 0) {
                        View f = mDialog.getCurrentFocus();
                        if (f != null) f.performClick();
                    }
                    return true;
                }
                case KeyEvent.KEYCODE_BUTTON_L1:
                    if (down) showTab(0);
                    return true;
                case KeyEvent.KEYCODE_BUTTON_R1:
                    if (down) showTab(1);
                    return true;
                case KeyEvent.KEYCODE_BUTTON_START:
                case KeyEvent.KEYCODE_BUTTON_SELECT:
                    return true;  // the - and + that opened the menu: nothing
                default:
                    return false;
            }
        });
        showTab(0);
        GameActivity.showImmersive(mDialog);
        if (w != null) {
            DisplayMetrics m = mA.getResources().getDisplayMetrics();
            w.setLayout(Math.round(m.widthPixels * 0.86f), Math.round(m.heightPixels * 0.92f));
        }
        mTabDriver.requestFocus();
    }

    private void showTab(int tab) {
        mTab = tab;
        mTabDriver.setSelected(tab == 0);
        mTabSettings.setSelected(tab == 1);
        mBody.removeAllViews();
        mValues = readSettings();
        if (tab == 0)
            buildDriverTab();
        else
            buildSettingsTab();
        mScroll.scrollTo(0, 0);
    }

    // ---- Driver tab ---------------------------------------------------------------------------------------------

    private static String driverLabel(String d) {
        return d == null || d.isEmpty() || d.equals("system") ? "System driver" : d;
    }

    private void buildDriverTab() {
        section("GPU driver");
        note("The Vulkan driver the game runs on. The phone's own (system) driver is the default. Others are "
                + "AdrenoTools driver packages (.zip, e.g. Mesa Turnip) or .so files in " + sDriversDir + ". "
                + "A new choice is remembered and used the next time the game starts.");
        note("In use now: " + driverLabel(sRunningDriver));

        String chosen = mValues.containsKey("gpu_driver") ? mValues.get("gpu_driver") : "system";
        if (chosen == null || chosen.isEmpty()) chosen = "system";
        List<String> drivers = new ArrayList<>();
        drivers.add("system");
        File dir = new File(sDriversDir);
        File[] files = dir.listFiles();
        if (files != null) {
            Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
            for (File f : files) {
                String n = f.getName().toLowerCase(Locale.ROOT);
                if (f.isFile() && (n.endsWith(".zip") || n.endsWith(".so"))) drivers.add(f.getName());
            }
        }
        if (!drivers.contains(chosen)) drivers.add(chosen);  // chosen, but the file is gone: still shown

        final List<TextView> rows = new ArrayList<>();
        for (final String d : drivers) {
            String label = d.equals("system") ? "System driver (default)" : d;
            if (d.equals(sRunningDriver)) label += "   • in use now";
            if (!d.equals("system") && !new File(dir, d).isFile()) label += "   • file missing";
            final TextView row = chip(label, false);
            row.setSelected(d.equals(chosen));
            row.setGravity(Gravity.CENTER_VERTICAL | Gravity.START);
            row.setOnClickListener(v -> {
                if (row.isSelected()) return;
                if (!writeSetting("gpu_driver", d)) {
                    popup("Could not save", "settings.txt could not be written:\n" + sSettingsPath);
                    return;
                }
                for (TextView r : rows) r.setSelected(r == row);
                mValues.put("gpu_driver", d);
                if (d.equals(sRunningDriver))
                    popup("Driver", driverLabel(d) + " is the driver in use now; it stays in use.");
                else
                    popup("Driver changed", driverLabel(d) + " will be used the next time you start the game.\n\n"
                            + "Close the app completely and open it again to switch now. If it cannot be loaded, the "
                            + "system driver is used.");
            });
            rows.add(row);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.topMargin = dp(6);
            mBody.addView(row, lp);
        }

        TextView add = chip("Add a driver file…", false);
        add.setGravity(Gravity.CENTER);
        add.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("*/*");
            try {
                mA.startActivityForResult(i, REQ_DRIVER);
            } catch (Exception e) {
                popup("No file picker", "Copy the driver file into " + sDriversDir + " instead.");
            }
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(14);
        mBody.addView(add, lp);
    }

    private void importDriver(Uri uri) {
        String name = null;
        try (Cursor c = mA.getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME}, null, null,
                null)) {
            if (c != null && c.moveToFirst()) name = c.getString(0);
        } catch (Exception ignored) {
        }
        if (name == null) name = uri.getLastPathSegment();
        if (name != null) name = name.substring(name.lastIndexOf('/') + 1);
        String lower = name == null ? "" : name.toLowerCase(Locale.ROOT);
        if (!lower.endsWith(".zip") && !lower.endsWith(".so")) {
            popup("Not a driver", "Choose a driver package (.zip, as made for AdrenoTools) or a .so file.");
            return;
        }
        File dir = new File(sDriversDir);
        dir.mkdirs();
        File out = new File(dir, name);
        try (InputStream in = mA.getContentResolver().openInputStream(uri);
                OutputStream o = new FileOutputStream(out)) {
            if (in == null) throw new IOException("cannot open the file");
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) o.write(buf, 0, n);
        } catch (IOException e) {
            out.delete();
            popup("Could not add the driver", "Copying " + name + " failed: " + e.getMessage());
            return;
        }
        if (mTab == 0) showTab(0);
        popup("Driver added", name + " was copied to the drivers folder. Choose it in the list to use it.");
    }

    // ---- Settings tab -------------------------------------------------------------------------------------------

    private void buildSettingsTab() {
        section("Display");
        choice("fps", "Frame rate cap", "Changes at once.", new String[] {"30", "45", "60"},
                new String[] {"30 fps", "45 fps", "60 fps"}, "60");
        choice("aspect_ratio", "Aspect ratio (experimental)", "16:9 is the game's own. Fill screen widens (or narrows) the view "
                + "to the screen's shape. Changes at the next start.",
                new String[] {"16:9", "auto", "4:3", "21:9"},
                new String[] {"16:9 (default)", "Fill screen", "4:3", "21:9"}, "16:9");
        toggle("fps_counter", "FPS counter",
                "Shows the frame rate at the top of the screen. Drawn by Android over the game, so it costs the game "
                        + "no frames. Changes at once.",
                "1", "0", "0");
        choice("exposure", "Brightness", "Changes at once.", new String[] {"1.0", "1.37", "1.7"},
                new String[] {"Original", "Default", "Brighter"}, "1.37");
        choice("present_mode", "Frame presentation", "Changes at the next start.",
                new String[] {"mailbox", "fifo"}, new String[] {"Low latency (default)", "V-sync"}, "mailbox");

        section("Controls");
        toggle("face_buttons", "Swap A / B and X / Y inputs",
                "On: the controller's face buttons act by position, as on a Switch (the right button confirms). "
                        + "Changes at once.",
                "position", "label", "label");
        choice("touch_controls", "Touch controls", "Changes at once.", new String[] {"1", "0"},
                new String[] {"On", "Off"}, "1");
        choice("touch_opacity", "Touch controls opacity", "Changes at once.",
                new String[] {"0.3", "0.5", "0.7", "1.0"}, new String[] {"30%", "50%", "70%", "100%"}, "0.7");
        choice("touch_camera_speed", "Touch camera speed", "Changes at once.",
                new String[] {"0.75", "1.0", "1.5", "2.0"}, new String[] {"Slow", "Normal", "Fast", "Faster"}, "1.0");
        toggle("touch_camera_recenter", "Prevent right stick centre drift",
                "On: the touch camera stick stays where it is while your thumb rests on it, so the camera keeps "
                        + "turning until you move back. Off: it drifts back to the centre under a resting thumb and the "
                        + "camera eases to a stop. Changes at once.",
                "0", "1", "1");
        choice("touch_camera_invert", "Invert touch camera", "Changes at once.", new String[] {"0", "1"},
                new String[] {"Off", "On"}, "0");

        section("Game folder");
        String custom = mValues.containsKey("game_dir") ? mValues.get("game_dir") : "";
        note("Game files in use: " + sGameDir);
        note("The app looks in " + sDefaultGameDir + " first. If the game is not there, it uses the folder chosen "
                + "here (or asks for one)." + (custom == null || custom.isEmpty() ? "" : "\nChosen folder: " + custom));
        TextView pick = chip("Choose another game folder…", false);
        pick.setGravity(Gravity.CENTER);
        pick.setOnClickListener(v -> {
            try {
                mA.startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_FOLDER);
            } catch (Exception e) {
                popup("No file manager", "No app is available to choose a folder.");
            }
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(8);
        mBody.addView(pick, lp);
        note("Settings are saved in " + sSettingsPath);
    }

    private void chooseGameFolder(Uri uri) {
        String path = treeToPath(uri);
        String root = findGameRoot(path);
        if (root == null) {
            popup("No game files there", (path == null ? "That folder" : path) + " does not hold the game files "
                    + "(no exefs/main in it). Choose the folder that contains the exefs folder and dvdbnd0.bdt.");
            return;
        }
        if (!writeSetting("game_dir", root)) {
            popup("Could not save", "settings.txt could not be written:\n" + sSettingsPath);
            return;
        }
        if (mTab == 1) showTab(1);
        popup("Game folder", "Game folder set to " + root + ".\n\nIt is used from the next start whenever the game "
                + "is not in " + sDefaultGameDir + ".");
    }

    // ---- building blocks ----------------------------------------------------------------------------------------

    private static boolean same(String a, String b) {
        if (a == null || b == null) return false;
        if (a.equalsIgnoreCase(b)) return true;
        try {
            return Math.abs(Float.parseFloat(a) - Float.parseFloat(b)) < 1e-3f;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /** A row of choices for one settings.txt key. */
    private void choice(final String key, final String title, String noteText, final String[] values,
            String[] labels, String def) {
        String cur = mValues.containsKey(key) ? mValues.get(key) : def;
        TextView t = text(title, 15, TEXT, true);
        t.setPadding(0, dp(10), 0, 0);
        mBody.addView(t);
        if (noteText != null) {
            TextView n = text(noteText, 12, DIM, false);
            mBody.addView(n);
        }
        HorizontalScrollView hs = new HorizontalScrollView(mA);
        hs.setHorizontalScrollBarEnabled(false);
        LinearLayout row = new LinearLayout(mA);
        row.setOrientation(LinearLayout.HORIZONTAL);
        hs.addView(row);
        final List<TextView> chips = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            final String value = values[i];
            final TextView c = chip(labels[i], true);
            c.setSelected(same(value, cur));
            c.setOnClickListener(v -> {
                if (c.isSelected()) return;
                if (!writeSetting(key, value)) {
                    popup("Could not save", "settings.txt could not be written:\n" + sSettingsPath);
                    return;
                }
                for (TextView o : chips) o.setSelected(o == c);
                mValues.put(key, value);
                applyChoice(key, title, value);
            });
            chips.add(c);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.rightMargin = dp(8);
            lp.topMargin = dp(6);
            row.addView(c, lp);
        }
        mBody.addView(hs);
    }

    /** An on / off switch for one settings.txt key. */
    private void toggle(final String key, final String title, String noteText, final String onValue,
            final String offValue, String def) {
        String cur = mValues.containsKey(key) ? mValues.get(key) : def;
        LinearLayout row = new LinearLayout(mA);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(14), dp(10), dp(14), dp(10));
        row.setFocusable(true);
        row.setClickable(true);
        StateListDrawable bg = new StateListDrawable();
        bg.addState(new int[] {android.R.attr.state_focused}, rounded(0xFF3A3A3A, 10, 0xFFFFFFFF));
        bg.addState(new int[] {android.R.attr.state_pressed}, rounded(0xFF3A3A3A, 10, 0));
        bg.addState(new int[] {}, rounded(CARD, 10, 0));
        row.setBackground(bg);
        LinearLayout texts = new LinearLayout(mA);
        texts.setOrientation(LinearLayout.VERTICAL);
        texts.addView(text(title, 15, TEXT, true));
        if (noteText != null) texts.addView(text(noteText, 12, DIM, false));
        row.addView(texts, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        final TextView state = new TextView(mA);
        state.setTextSize(14);
        state.setTypeface(Typeface.DEFAULT_BOLD);
        state.setGravity(Gravity.CENTER);
        state.setMinWidth(dp(64));
        state.setPadding(dp(12), dp(6), dp(12), dp(6));
        state.setDuplicateParentStateEnabled(false);
        final boolean[] on = {same(cur, onValue)};
        final Runnable paint = () -> {
            state.setText(on[0] ? "On" : "Off");
            state.setTextColor(on[0] ? 0xFF151515 : TEXT);
            state.setBackground(rounded(on[0] ? GOLD : 0xFF444444, 14, 0));
        };
        paint.run();
        LinearLayout.LayoutParams slp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        slp.leftMargin = dp(12);
        row.addView(state, slp);
        row.setOnClickListener(v -> {
            String value = on[0] ? offValue : onValue;
            if (!writeSetting(key, value)) {
                popup("Could not save", "settings.txt could not be written:\n" + sSettingsPath);
                return;
            }
            on[0] = !on[0];
            paint.run();
            mValues.put(key, value);
            applyChoice(key, title, value);
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(10);
        mBody.addView(row, lp);
    }

    private void applyChoice(String key, String title, String value) {
        boolean now = false;
        try {
            now = nativeApply(key, value);
        } catch (Throwable e) {
            android.util.Log.w("DSRNative", "settings menu: nativeApply failed", e);
        }
        if (key.equals("fps_counter")) {
            FpsCounter.apply(mA, "1".equals(value));
            now = true;
        } else if (key.equals("touch_camera_recenter")) {
            TouchControls.sRecenter = !"0".equals(value);
            now = true;
        } else if (key.startsWith("touch_") && !key.equals("touch_camera_invert")) {
            boolean on = !"0".equals(mValues.get("touch_controls"));
            TouchControls.applySettings(mA, on, parse(mValues.get("touch_opacity"), 0.7f),
                    parse(mValues.get("touch_camera_speed"), 1.0f));
            now = true;
        }
        if (now)
            Toast.makeText(mA, title + ": applied", Toast.LENGTH_SHORT).show();
        else
            popup(title, title + " will change the next time you start the game.");
    }

    private static float parse(String s, float def) {
        try {
            return s == null ? def : Float.parseFloat(s);
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private void section(String title) {
        TextView t = text(title.toUpperCase(Locale.ROOT), 13, GOLD, true);
        t.setLetterSpacing(0.08f);
        t.setPadding(0, dp(mBody.getChildCount() == 0 ? 2 : 18), 0, dp(2));
        mBody.addView(t);
    }

    private void note(String s) {
        TextView n = text(s, 12, DIM, false);
        n.setPadding(0, dp(4), 0, 0);
        mBody.addView(n);
    }

    private TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(mA);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private GradientDrawable rounded(int fill, float radiusDp, int stroke) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(fill);
        g.setCornerRadius(dp(radiusDp));
        if (stroke != 0) g.setStroke(dp(1.5f), stroke);
        return g;
    }

    /** A focusable, clickable choice: gold when chosen, outlined when it has the controller's focus. */
    private TextView chip(String label, boolean compact) {
        TextView c = new TextView(mA);
        c.setText(label);
        c.setTextSize(compact ? 14 : 15);
        c.setPadding(dp(compact ? 14 : 16), dp(compact ? 8 : 12), dp(compact ? 14 : 16), dp(compact ? 8 : 12));
        c.setFocusable(true);
        c.setClickable(true);
        StateListDrawable bg = new StateListDrawable();
        bg.addState(new int[] {android.R.attr.state_selected, android.R.attr.state_focused}, rounded(GOLD, 10, 0xFFFFFFFF));
        bg.addState(new int[] {android.R.attr.state_selected}, rounded(GOLD, 10, 0));
        bg.addState(new int[] {android.R.attr.state_focused}, rounded(0xFF3A3A3A, 10, 0xFFFFFFFF));
        bg.addState(new int[] {android.R.attr.state_pressed}, rounded(0xFF3A3A3A, 10, 0));
        bg.addState(new int[] {}, rounded(CARD, 10, 0));
        c.setBackground(bg);
        c.setTextColor(new ColorStateList(new int[][] {{android.R.attr.state_selected}, {}},
                new int[] {0xFF151515, TEXT}));
        if (compact) {
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.leftMargin = dp(8);
            c.setLayoutParams(lp);
        }
        return c;
    }

    private void popup(String title, String message) {
        AlertDialog d = new AlertDialog.Builder(mA, android.R.style.Theme_DeviceDefault_Dialog_Alert)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton(android.R.string.ok, null)
                .create();
        GameActivity.showImmersive(d);
    }
}
