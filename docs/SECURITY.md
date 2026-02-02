# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in NanoDistill, please report it responsibly and do not publicly disclose the vulnerability until we have had a chance to address it.

### Reporting Process

1. **Do NOT open a public GitHub issue** for security vulnerabilities

2. **Email the maintainers** at: security@github.com (replace with actual contact if you set one up)
   - Include a clear description of the vulnerability
   - Steps to reproduce (if applicable)
   - Potential impact
   - Suggested fix (if you have one)

3. **Wait for acknowledgment** - We aim to acknowledge vulnerability reports within 48 hours

4. **Coordinate disclosure** - We'll work with you to:
   - Understand and confirm the vulnerability
   - Develop and test a fix
   - Plan a responsible disclosure timeline

### Security Considerations

**API Key Handling:**
- Never commit API keys (ANTHROPIC_API_KEY, etc.) to the repository
- Use environment variables via `.env` files (which should be in `.gitignore`)
- The `.gitignore` already excludes `.env` files

**Data Privacy:**
- Seed data and training outputs should be treated as potentially sensitive
- Users are responsible for secure handling of their data
- Consider your API provider's data retention policies

**Dependency Management:**
- We regularly update dependencies for security patches
- Users should keep their installations up to date

## Security Best Practices for Users

When using NanoDistill:

1. **Keep your API keys secret** - Don't share or commit them
2. **Review seed data** - Understand what data you're sending to the teacher model
3. **Update regularly** - Install security updates: `pip install --upgrade nanodistill`
4. **Review outputs** - Inspect generated models and synthetic data before using in production
5. **Monitor API usage** - Check your API provider's dashboard for unusual activity

## Supported Versions

We provide security updates for:

- Current version: security updates and patches
- Previous version: critical security patches only
- Older versions: no guarantee of updates

## Known Issues

None currently reported. Any known security issues will be documented here.

## Contact

For security questions or concerns, please reach out to the maintainers directly rather than posting in public issues.

---

Thank you for helping keep NanoDistill secure!
