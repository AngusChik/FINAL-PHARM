Option Explicit

If WScript.Arguments.Count = 1 Then
    If LCase(WScript.Arguments.Item(0)) = "--probe" Then
        WScript.Quit 0
    End If
End If

Dim shell, fileSystem, scriptDirectory, runnerPath, powerShellPath, command, exitCode, runnerArguments
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
runnerPath = fileSystem.BuildPath(scriptDirectory, "run-scheduled-jobs.ps1")
powerShellPath = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe")
If Not fileSystem.FileExists(runnerPath) Or Not fileSystem.FileExists(powerShellPath) Then
    WScript.Quit 2
End If
runnerArguments = ""
If WScript.Arguments.Count = 1 Then
    If LCase(WScript.Arguments.Item(0)) = "--self-test" Then
        runnerArguments = " -SelfTest"
    End If
End If
command = QuoteArgument(powerShellPath) & " -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & QuoteArgument(runnerPath) & runnerArguments

' Window style 0 prevents creation of a visible console. Waiting preserves the
' PowerShell runner's exit code for Task Scheduler and its failure history.
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode

Function QuoteArgument(value)
    QuoteArgument = Chr(34) & Replace(value, Chr(34), Chr(34) & Chr(34)) & Chr(34)
End Function
