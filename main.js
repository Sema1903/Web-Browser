const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
let mainWindow;
function createWindow() {
 mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
    preload: path.join(__dirname, "preload.js"),
    contextIsolation: true,
    enableRemoteModule: false,
    },
});
mainWindow.loadFile("index.html");
mainWindow.once("ready-to-show", () => {
    mainWindow.show();
});
ipcMain.on("open-url", (event, url) => {
    mainWindow.loadURL(url);
  });
}
app.on("ready", createWindow);
app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});
app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});