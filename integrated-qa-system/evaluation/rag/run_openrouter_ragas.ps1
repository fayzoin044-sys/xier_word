param(
    [switch]$Full
)

$ErrorActionPreference = 'Stop'

$pythonPath = (Get-Command python -ErrorAction Stop).Source
$evaluationScript = Join-Path $PSScriptRoot 'evaluate_ragas_gpt.py'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $evaluationScript -PathType Leaf)) {
    throw "Evaluation script not found: $evaluationScript"
}

Write-Host ''
Write-Host 'Enter your OpenRouter API Key, then press Enter.' -ForegroundColor Cyan
Write-Host 'No characters will be displayed.' -ForegroundColor Yellow

$secureKey = Read-Host 'OpenRouter API Key' -AsSecureString
$plainKey = [System.Net.NetworkCredential]::new('', $secureKey).Password

if ([string]::IsNullOrWhiteSpace($plainKey)) {
    throw 'No API Key was entered.'
}
if (-not $plainKey.StartsWith('sk-or-')) {
    throw 'This does not look like an OpenRouter API Key (expected prefix: sk-or-).'
}

try {
    $env:OPENROUTER_API_KEY = $plainKey
    Write-Host ''
    $evaluationArguments = @(
        $evaluationScript,
        '--api-provider',
        'openrouter',
        '--execute'
    )
    if ($Full) {
        $evaluationArguments += @('--limit', '0')
        Write-Host 'Starting the full 29-case GPT + Ragas evaluation...' -ForegroundColor Green
    }
    else {
        Write-Host 'Starting the 3-case GPT + Ragas smoke test...' -ForegroundColor Green
    }
    & $pythonPath @evaluationArguments
    $evaluationExitCode = $LASTEXITCODE
}
finally {
    Remove-Item Env:OPENROUTER_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    $secureKey = $null
}

if ($evaluationExitCode -ne 0) {
    throw "Evaluation process exit code: $evaluationExitCode"
}

Write-Host ''
Write-Host 'The evaluation is complete.' -ForegroundColor Green
