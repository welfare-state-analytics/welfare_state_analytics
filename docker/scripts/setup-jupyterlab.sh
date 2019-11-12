#!/bin/bash

SETTING_PATH=${HOME}/.jupyter/lab/user-settings/@jupyterlab

mkdir -p ${SETTING_PATH}/extensionmanager-extension
cat > ${SETTING_PATH}/extensionmanager-extension/plugin.jupyterlab-setting <<EOD
{
    // Extension Manager
    // @jupyterlab/extensionmanager-extension:plugin
    // Extension manager settings.
    // *********************************************

    // Enabled Status
    // Enables extension manager (requires Node.js/npm).
    // WARNING: installing untrusted extensions may be unsafe.
    "enabled": true
}
EOD

mkdir -p ${SETTING_PATH}/notebook-extension
cat > ${SETTING_PATH}/notebook-extension/tracker.jupyterlab-settings <<EOD
{
    // Notebook
    // @jupyterlab/notebook-extension:tracker
    // Notebook settings.
    // **************************************

    // Code Cell Configuration
    // The configuration for all code cells.
    "codeCellConfig": {
        "autoClosingBrackets": true,
        "fontFamily": null,
        "fontSize": null,
        "lineHeight": null,
        "lineNumbers": false,
        "lineWrap": "off",
        "matchBrackets": true,
        "readOnly": false,
        "insertSpaces": true,
        "tabSize": 4,
        "wordWrapColumn": 80,
        "rulers": [],
        "codeFolding": false
    },

    // Default cell type
    // The default type (markdown, code, or raw) for new cells
    "defaultCell": "code",

    // Shut down kernel
    // Whether to shut down or not the kernel when closing a notebook.
    "kernelShutdown": false,

    // Markdown Cell Configuration
    // The configuration for all markdown cells.
    "markdownCellConfig": {
        "autoClosingBrackets": false,
        "fontFamily": null,
        "fontSize": null,
        "lineHeight": null,
        "lineNumbers": false,
        "lineWrap": "on",
        "matchBrackets": false,
        "readOnly": false,
        "insertSpaces": true,
        "tabSize": 4,
        "wordWrapColumn": 80,
        "rulers": [],
        "codeFolding": false
    },

    // Raw Cell Configuration
    // The configuration for all raw cells.
    "rawCellConfig": {
        "autoClosingBrackets": false,
        "fontFamily": null,
        "fontSize": null,
        "lineHeight": null,
        "lineNumbers": false,
        "lineWrap": "on",
        "matchBrackets": false,
        "readOnly": false,
        "insertSpaces": true,
        "tabSize": 4,
        "wordWrapColumn": 80,
        "rulers": [],
        "codeFolding": false
    },

    // Scroll past last cell
    // Whether to be able to scroll so the last cell is at the top of the panel
    "scrollPastEnd": true
}
EOD

