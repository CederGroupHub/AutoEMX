#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .legacy_standards import (
    is_legacy_standards_payload,
    migrate_legacy_standards_payload,
    normalize_standards_file_payload,
)

__all__ = [
    "is_legacy_standards_payload",
    "migrate_legacy_standards_payload",
    "normalize_standards_file_payload",
]
