"""Permissioned action registry for the desktop apps Voice2Text AI can talk about.

Part 6 of the Ollama onboarding plan (issue #23). Nothing here launches, installs
or configures anything: the registry only names what *could* be allowed later,
and the settings list says which of those names the user has actually granted.
Availability is read from Flatpak metadata and ``.desktop`` files, both of which
are plain data on disk.

The permission string on each action follows the app's own sandbox vocabulary
(``io.github.crhy.voice2textai.yml`` ``finish-args:``): an action that touches a
folder has to declare the bind it needs, exactly like the manifest does. Read-only
actions get ``--ro-bind``, export actions get ``--bind`` on a folder they may
create files in, and every action that involves the other app declares the
host-spawn permission the manifest already carries.

Deny by default: an action the user has not named by id is not available, and an
id that is not in the registry is never accepted, however it is spelled.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# The four apps the issue names. Availability is checked for these and only
# these; anything else installed on the desktop stays out of the registry.
APP_IDS: tuple[str, ...] = (
    "org.gimp.GIMP",
    "org.blender.Blender",
    "org.freecad.FreeCAD",
    "org.libreoffice.LibreOffice",
)

APP_NAMES: dict[str, str] = {
    "org.gimp.GIMP": "GIMP",
    "org.blender.Blender": "Blender",
    "org.freecad.FreeCAD": "FreeCAD",
    "org.libreoffice.LibreOffice": "LibreOffice",
}

# Every action that involves the other app has to start it, which for a Flatpak
# means the host-spawn permission the manifest already declares. Folder access
# is the part the user has to choose, so it is spelled as the bind flag itself.
LAUNCH_PERMISSION = "--talk-name=org.freedesktop.Flatpak"


def _permission(*flags: str) -> str:
    return " ".join((LAUNCH_PERMISSION, *flags))


@dataclass(slots=True, frozen=True)
class Action:
    app_id: str
    action_id: str
    title: str
    detail: str
    permission: str
    destructive: bool = False


# Read-only or export-only actions. Nothing here edits a project in place: the
# worst an export action may do is replace a file the user already named, and
# those are the only ones flagged ``destructive`` so the app can ask first.
ACTIONS: tuple[Action, ...] = (
    Action(
        "org.gimp.GIMP",
        "gimp.open-project",
        "Open this project file in GIMP",
        "Opens the image or .xcf you name so GIMP shows it while you dictate. "
        "GIMP only reads the file; nothing is written back.",
        _permission("--ro-bind=<GIMP project folder>"),
    ),
    Action(
        "org.gimp.GIMP",
        "gimp.export-caption",
        "Write the transcript as a new caption file",
        "Saves the transcript next to the image as a caption file that does not "
        "exist yet, so GIMP can place it later.",
        _permission("--bind=<GIMP export folder>"),
    ),
    Action(
        "org.gimp.GIMP",
        "gimp.overwrite-caption",
        "Replace an existing GIMP caption file",
        "Overwrites a caption file that is already in the export folder. The old "
        "text is gone afterwards, which is why this one is asked about first.",
        _permission("--bind=<GIMP export folder>"),
        destructive=True,
    ),
    Action(
        "org.blender.Blender",
        "blender.open-project",
        "Open this .blend file in Blender",
        "Opens the .blend you name so Blender shows the scene while you dictate. "
        "The file is only read.",
        _permission("--ro-bind=<Blender project folder>"),
    ),
    Action(
        "org.blender.Blender",
        "blender.list-project-files",
        "List the .blend files in a folder",
        "Reads a folder's names so you can dictate about the right scene without "
        "opening any of them.",
        _permission("--ro-bind=<Blender project folder>"),
    ),
    Action(
        "org.blender.Blender",
        "blender.export-notes",
        "Write the transcript as new notes beside the scene",
        "Saves the transcript as a new text file in the project's blend files "
        "folder, so the notes travel with the scene.",
        _permission("--bind=<Blender notes folder>"),
    ),
    Action(
        "org.freecad.FreeCAD",
        "freecad.open-project",
        "Open this .FCStd file in FreeCAD",
        "Opens the FreeCAD document you name so it is on screen while you dictate. "
        "The file is only read.",
        _permission("--ro-bind=<FreeCAD project folder>"),
    ),
    Action(
        "org.freecad.FreeCAD",
        "freecad.export-model",
        "Export the open model to a new STEP file",
        "Writes the model you are looking at to a STEP file that does not exist "
        "yet, so the geometry leaves the app without the app being touched again.",
        _permission("--bind=<FreeCAD export folder>"),
    ),
    Action(
        "org.freecad.FreeCAD",
        "freecad.export-transcript",
        "Write the transcript as a new file in the project folder",
        "Saves the transcript as a new text file beside the .FCStd, so the "
        "dictation and the model stay together.",
        _permission("--bind=<FreeCAD project folder>"),
    ),
    Action(
        "org.libreoffice.LibreOffice",
        "libreoffice.open-document",
        "Open this document in LibreOffice",
        "Opens the document you name so LibreOffice shows it while you dictate. "
        "The file is only read.",
        _permission("--ro-bind=<document folder>"),
    ),
    Action(
        "org.libreoffice.LibreOffice",
        "libreoffice.send-transcript",
        "Send the transcript to LibreOffice as a new document",
        "Saves the transcript as a new .odt you name, so it can be edited and "
        "printed from LibreOffice afterwards.",
        _permission("--bind=<document folder>"),
    ),
    Action(
        "org.libreoffice.LibreOffice",
        "libreoffice.overwrite-document",
        "Replace an existing LibreOffice document with the transcript",
        "Overwrites a document that already exists. Whatever was in it is gone "
        "afterwards, so this is confirmed before it is ever allowed.",
        _permission("--bind=<document folder>"),
        destructive=True,
    ),
)

_BY_ID: dict[str, Action] = {action.action_id: action for action in ACTIONS}
ACTION_IDS: frozenset[str] = frozenset(_BY_ID)


class SettingsLike(Protocol):
    """Anything with an ``allowed_actions`` list; ``config.Settings`` satisfies it."""

    allowed_actions: list[str]


@dataclass(slots=True, frozen=True)
class AppInstall:
    app_id: str
    name: str
    source: str  # "flatpak" or "desktop"


def _flatpak_applications() -> set[str]:
    """Application ids Flatpak already knows about, or an empty set.

    ``flatpak list`` only reads metadata that is already on disk: it never
    launches, installs or updates anything. A missing Flatpak, or one that
    answers nothing, simply means "fall back to the .desktop files".
    """
    try:
        result = subprocess.run(
            ["flatpak", "list", "--columns=application"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return _parse_flatpak_list(result.stdout)


def _parse_flatpak_list(output: str) -> set[str]:
    return {line.strip() for line in output.splitlines() if line.strip()}


def _desktop_dirs() -> tuple[Path, ...]:
    # The two application directories any non-Flatpak package installs into,
    # plus the Flatpak export trees: a Flatpak-installed app's .desktop file
    # lives there rather than in ~/.local/share/applications, so the fallback
    # has to look there too or it reports nothing on exactly the machines
    # this app ships to.
    return (
        Path("/usr/share/applications"),
        Path.home() / ".local/share/applications",
        Path("/var/lib/flatpak/exports/share/applications"),
        Path.home() / ".local/share/flatpak/exports/share/applications",
    )


def available_apps(
    *,
    flatpak_output: str | None = None,
    desktop_dirs: Sequence[Path] | None = None,
) -> list[AppInstall]:
    """Which of the four registry apps are installed, without launching anything.

    ``flatpak_output`` lets a caller (or a test) hand over the text of
    ``flatpak list --columns=application`` instead of running it; ``desktop_dirs``
    replaces the searched application directories.
    """
    installed_ids = _parse_flatpak_list(flatpak_output) if flatpak_output is not None else _flatpak_applications()
    dirs = tuple(desktop_dirs) if desktop_dirs is not None else _desktop_dirs()

    found: list[AppInstall] = []
    for app_id in APP_IDS:
        if app_id in installed_ids:
            found.append(AppInstall(app_id, APP_NAMES[app_id], "flatpak"))
            continue
        for directory in dirs:
            # Exact file name only: org.libreoffice.LibreOffice.base.desktop
            # is a different app's launcher, not evidence of LibreOffice's
            # registry entry, and org.blender.Blender.Codecs is a module.
            if (directory / f"{app_id}.desktop").is_file():
                found.append(AppInstall(app_id, APP_NAMES[app_id], "desktop"))
                break
    return found


def normalize_ids(values: object) -> list[str]:
    """Keep only ids this registry knows, deduplicated and sorted.

    A hand-edited or older config.json can hold anything under this key, so a
    value that is not a list of strings is treated as "nothing granted".
    """
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        return []
    clean = {item.strip() for item in values if isinstance(item, str)}
    return sorted(clean & ACTION_IDS)


def allowed_ids(settings: SettingsLike) -> tuple[str, ...]:
    """The granted ids, in registry order, unknown ones dropped."""
    return tuple(normalize_ids(getattr(settings, "allowed_actions", ())))


def allowed(settings: SettingsLike) -> list[Action]:
    """The actions the user has named, sorted by id. Empty unless granted."""
    return [_BY_ID[action_id] for action_id in allowed_ids(settings)]


def granted(settings: SettingsLike, action_id: str) -> Action | None:
    """The action behind a granted id, or None. Unknown ids are never accepted."""
    if action_id not in ACTION_IDS:
        return None
    if action_id not in allowed_ids(settings):
        return None
    return _BY_ID[action_id]


def grant(settings: SettingsLike, action_id: str) -> bool:
    """Name an action in the settings. Refuses ids the registry does not know."""
    if action_id not in ACTION_IDS:
        return False
    settings.allowed_actions = normalize_ids([*settings.allowed_actions, action_id])
    return True


def revoke(settings: SettingsLike, action_id: str) -> bool:
    """Drop a previously named action."""
    if action_id not in ACTION_IDS:
        return False
    settings.allowed_actions = normalize_ids(
        [item for item in settings.allowed_actions if item.strip() != action_id]
    )
    return True


def actions_for(app_id: str) -> list[Action]:
    """Every action belonging to one app, granted or not."""
    return [action for action in ACTIONS if action.app_id == app_id]


def availability_text(apps: Sequence[AppInstall]) -> str:
    return ", ".join(app.name for app in apps) or "none detected"


def granted_text(settings: SettingsLike) -> str:
    return ", ".join(action.action_id for action in allowed(settings)) or "none granted"


def describe(action: Action) -> str:
    """One line for the status area's tooltip: what it does and what it needs."""
    flag = "destructive — " if action.destructive else ""
    return f"{action.title} [{action.app_id}] {flag}{action.detail} needs {action.permission}"
