"""Credential redaction shared by persisted records and reports."""
import re

def redact_sensitive(value):
    if isinstance(value, dict):
        return {k: ('[redacted]' if re.search(r'password|token|secret', str(k), re.I) else redact_sensitive(v)) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)(rtsp[s]?|https?)://[^\s/@]+(?::[^\s/@]*)?@', r'\1://[redacted]@', value)
        return re.sub(r'(?i)([?&](?:password|passwd|token|api_key|secret)=)[^&\s]*', r'\1[redacted]', value)
    return value
