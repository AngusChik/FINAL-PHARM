Option Explicit

Dim shell, fileSystem, scriptDirectory, productionScript, powerShellPath
Dim command, exitCode, argument, noBrowser, quiet, userRequested

Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
productionScript = fileSystem.BuildPath(scriptDirectory, "production.ps1")
powerShellPath = shell.ExpandEnvironmentStrings( _
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" _
)

If Not fileSystem.FileExists(productionScript) Or _
   Not fileSystem.FileExists(powerShellPath) Then
    WScript.Quit 2
End If

If WScript.Arguments.Count = 1 Then
    If LCase(WScript.Arguments.Item(0)) = "--probe" Then
        WScript.Quit 0
    End If
End If

noBrowser = False
quiet = False
userRequested = False
For Each argument In WScript.Arguments
    Select Case LCase(argument)
        Case "--no-browser"
            noBrowser = True
        Case "--quiet"
            quiet = True
        Case "--user-requested"
            userRequested = True
        Case Else
            WScript.Quit 2
    End Select
Next

command = QuoteArgument(powerShellPath) & _
    " -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass" & _
    " -File " & QuoteArgument(productionScript) & _
    " -Action ensure -NonInteractive"
If noBrowser Then
    command = command & " -NoBrowser"
End If
If userRequested Then
    command = command & " -UserRequested"
End If

' Window style 0 avoids creating a command window. Waiting lets a desktop
' shortcut report startup failures while the server processes remain detached.
exitCode = shell.Run(command, 0, True)
If exitCode <> 0 And Not quiet Then
    shell.Popup _
        "Pharmacy could not start. Open Pharmacy Admin Control and review " & _
        "logs\production-control.log.", _
        0, "Pharmacy Startup", 16
End If
WScript.Quit exitCode

Function QuoteArgument(value)
    QuoteArgument = Chr(34) & Replace(value, Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
