"""Client-facing report generation.

``pdf.py`` holds presentation primitives (a thin, business-readable wrapper over
reportlab); the report modules alongside it hold *content* and are the only
place project findings are worded. Keeping the two apart means a new report is
a content module, not a new PDF engine.
"""
