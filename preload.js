const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("electronAPI", {
    openURL: (url) => ipcRenderer.send("open-url", url),
});