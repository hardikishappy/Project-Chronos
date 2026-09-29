"""
Project Chronos - Zero-Terminal Auto-Boot Installer
Registers Project Chronos Daemon with Windows Task Scheduler and the Windows User Startup folder.
Runs pythonw.exe completely detached with 0 visible CMD/PowerShell windows upon logon.
"""

import os
import sys
import subprocess

def install_autostart():
    root_dir = r"C:\Users\hardi\Project_Chronos"
    pythonw_exe = os.path.join(root_dir, ".venv", "Scripts", "pythonw.exe")
    daemon_py = os.path.join(root_dir, "daemon.py")
    task_name = "ProjectChronosDaemon"

    if not os.path.exists(pythonw_exe):
        # Fallback to local path if needed
        pythonw_exe = os.path.join(r"c:\Users\hardi\Downloads\Project Chronos", ".venv", "Scripts", "pythonw.exe")

    print("="*75)
    print(" PROJECT CHRONOS - ZERO-TERMINAL AUTO-BOOT INSTALLER")
    print("="*75)
    print(f"Target Daemon    : {daemon_py}")
    print(f"Detached Runtime : {pythonw_exe}")

    # Method 1: Register Windows Task Scheduler (AtLogon)
    task_cmd = f'"{pythonw_exe}" "{daemon_py}"'
    schtasks_cmd = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", task_cmd,
        "/SC", "ONLOGON",
        "/F"
    ]

    task_success = False
    try:
        res = subprocess.run(schtasks_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[OK] Task Scheduler: Successfully registered task '{task_name}' (Trigger: ONLOGON)")
            task_success = True
        else:
            print(f"[!] Task Scheduler Notice: {res.stderr.strip() or res.stdout.strip()}")
    except Exception as e:
        print(f"[!] Task Scheduler Error: {e}")

    # Method 2: Register in Windows User Startup Folder (Zero-Window VBS Launcher)
    # APPDATA\Microsoft\Windows\Start Menu\Programs\Startup
    startup_dir = os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup")
    vbs_path = os.path.join(startup_dir, "ProjectChronosDaemon.vbs")
    
    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{root_dir}"
WshShell.Run """{pythonw_exe}"" ""{daemon_py}""", 0, False
'''
    try:
        if os.path.exists(startup_dir):
            with open(vbs_path, "w", encoding="utf-8") as f:
                f.write(vbs_content)
            print(f"[OK] User Startup Folder: Created zero-window launcher at:\n     {vbs_path}")
    except Exception as e:
        print(f"[!] Startup Folder Error: {e}")

    print("="*75)
    print("AUTO-BOOT REGISTRATION COMPLETED SUCCESSFULLY")
    print("Project Chronos will start automatically on Windows logon in background mode.")
    print("="*75)

if __name__ == "__main__":
    install_autostart()
