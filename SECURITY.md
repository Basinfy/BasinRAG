# Security Policy

## Supported Versions

We actively provide security updates for the current major/minor release:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0.0 | :x:                |

---

## Reporting a Vulnerability

The BasinRAG security team takes all vulnerabilities seriously. We appreciate your efforts to responsibly disclose findings.

If you believe you have found a security vulnerability in BasinRAG:

1. **Do NOT report security vulnerabilities through public GitHub issues.**
2. Please report your findings privately via GitHub Security Advisories:
   - Navigate to the **Security** tab of this repository.
   - Click **Report a vulnerability**.
3. Alternatively, you may email **security@basinfy.org** (or contact the repository maintainers).

### What to Include in Your Report
- A description of the vulnerability and its potential impact.
- Step-by-step instructions to reproduce the issue (proof of concept script or curl command).
- The version of BasinRAG and environment (OS, Python version) tested.
- Any suggested mitigations or patches if available.

### Response Timeline
- **Initial Acknowledgement**: Within 48 hours.
- **Triage & Severity Assessment**: Within 5 business days.
- **Fix & Public Advisory**: Coordinated release according to severity.

---

## Deployment Security Guidelines

When deploying BasinRAG in production environments:
1. **API Authentication**: Set `BASINRAG_API_KEY` to enforce bearer token authentication on REST and WebSocket endpoints.
2. **CORS Restrictions**: Do not allow wildcard `*` origins in production; explicitly configure trusted frontend domains.
3. **Storage Directory**: Ensure the `.basinrag` directory has restricted filesystem permissions (`chmod 700`).
4. **LLM Credentials**: Never commit API keys or provider tokens into repositories; utilize environment variables.