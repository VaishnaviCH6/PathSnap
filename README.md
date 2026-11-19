# PathSnap M4 map display sync fix

This frontend-only overlay fixes a stale-map mismatch in which the trip dropdown
shows the newly requested trip while the iframe still displays the previously
reconstructed trip during an in-flight request.

It does not change ML/HMM predictions, road geometry, trained artifacts, or
accuracy results.

To install in PowerShell from the repo root (after backing up local changes):

    Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\PathSnap_M4_Map_Display_Fix.zip" -DestinationPath "D:\PathSnap" -Force

Restart the frontend Uvicorn process and hard-refresh in the browser.

This fix hides any prior map when a different trip is selected, disables the
trip dropdown during reconstruction, shows a loading placeholder, and ensures
that a failed request never leaves an old map visible under the new trip ID.

Sharp detours in a fully loaded, correct-trip route are a separate ML/HMM or
road-network issue and must be evaluated using reference-route metrics.
