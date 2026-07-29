# Voice2Text AI microphone dropdown readability fix

This overlay updates the existing `release/v0.4.0-native-linux` branch.

Changes:

- Widens the Preferences dialog.
- Shows the complete selected microphone name as the row subtitle and tooltip.
- Uses a wider selected-value label with middle ellipsis only when unavoidable.
- Wraps every microphone source name in the dropdown list, so long PipeWire,
  ALSA, USB, monitor, and hardware source names can be read in full.
- Does not alter dictation, rendering, TTS, Flatpak dependencies, or settings.
