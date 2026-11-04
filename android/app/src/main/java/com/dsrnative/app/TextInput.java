package com.dsrnative.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.text.InputFilter;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;

/**
 * Text entry for the game's software keyboard (nn::swkbd): shows a dialog with the system keyboard and
 * hands the result back to native code. Called from a native thread; the dialog runs on the UI thread.
 */
public final class TextInput {
    private TextInput() {}

    /** Delivered once per show(): ok = false when the player cancels. */
    static native void onResult(String text, boolean ok);

    public static void show(final Activity activity, final String title, final String initial, final int maxLength) {
        activity.runOnUiThread(() -> {
            final EditText edit = new EditText(activity);
            edit.setSingleLine(true);
            edit.setImeOptions(EditorInfo.IME_ACTION_DONE);
            if (maxLength > 0) edit.setFilters(new InputFilter[] {new InputFilter.LengthFilter(maxLength)});
            edit.setText(initial);
            edit.setSelection(edit.getText().length());
            final AlertDialog dialog = new AlertDialog.Builder(activity)
                    .setTitle(title)
                    .setView(edit)
                    .setCancelable(false)
                    .setPositiveButton(android.R.string.ok, (d, w) -> onResult(edit.getText().toString(), true))
                    .setNegativeButton(android.R.string.cancel, (d, w) -> onResult("", false))
                    .create();
            edit.setOnEditorActionListener((v, action, event) -> {
                if (action == EditorInfo.IME_ACTION_DONE) {
                    dialog.dismiss();
                    onResult(edit.getText().toString(), true);
                    return true;
                }
                return false;
            });
            dialog.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE);
            dialog.show();
            edit.requestFocus();
            InputMethodManager imm = (InputMethodManager) activity.getSystemService(Context.INPUT_METHOD_SERVICE);
            if (imm != null) imm.showSoftInput(edit, InputMethodManager.SHOW_IMPLICIT);
        });
    }
}
