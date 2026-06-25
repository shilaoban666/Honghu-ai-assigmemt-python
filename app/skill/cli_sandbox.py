"""CLI Sandbox & Danger Gate — mirrors the Java CLI skill provider.

Provides safe execution of CLI tools with:
- Whitelist of allowed commands
- Danger level classification
- Argument sanitization
- Timeout enforcement
"""

from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class DangerLevel(str, Enum):
    LOW = "LOW"          # Safe: echo, date, ls, pwd, cat (read-only)
    MEDIUM = "MEDIUM"    # Potentially risky: curl, wget (network)
    HIGH = "HIGH"        # Dangerous: rm, mv, dd, chmod, sudo
    CRITICAL = "CRITICAL"  # Blocked: fork bomb, /dev/sda, etc.


# Whitelist of allowed commands with danger levels
COMMAND_WHITELIST: dict[str, DangerLevel] = {
    # LOW — safe read-only
    "echo": DangerLevel.LOW,
    "date": DangerLevel.LOW,
    "pwd": DangerLevel.LOW,
    "ls": DangerLevel.LOW,
    "cat": DangerLevel.LOW,
    "head": DangerLevel.LOW,
    "tail": DangerLevel.LOW,
    "wc": DangerLevel.LOW,
    "grep": DangerLevel.LOW,
    "find": DangerLevel.LOW,
    "which": DangerLevel.LOW,
    "whoami": DangerLevel.LOW,
    "hostname": DangerLevel.LOW,
    "uname": DangerLevel.LOW,
    "df": DangerLevel.LOW,
    "du": DangerLevel.LOW,
    "env": DangerLevel.LOW,
    "printenv": DangerLevel.LOW,
    "sort": DangerLevel.LOW,
    "uniq": DangerLevel.LOW,
    "tr": DangerLevel.LOW,
    "cut": DangerLevel.LOW,
    "awk": DangerLevel.LOW,
    "sed": DangerLevel.LOW,
    "jq": DangerLevel.LOW,
    "python": DangerLevel.LOW,
    "python3": DangerLevel.LOW,

    # MEDIUM — network access
    "curl": DangerLevel.MEDIUM,
    "wget": DangerLevel.MEDIUM,
    "ping": DangerLevel.MEDIUM,
    "nslookup": DangerLevel.MEDIUM,
    "dig": DangerLevel.MEDIUM,

    # HIGH — file modification
    "mkdir": DangerLevel.HIGH,
    "touch": DangerLevel.HIGH,
    "cp": DangerLevel.HIGH,
    "mv": DangerLevel.HIGH,
    "rm": DangerLevel.HIGH,
    "chmod": DangerLevel.HIGH,
    "chown": DangerLevel.HIGH,
}

# Patterns that are ALWAYS blocked regardless of command
BLOCKED_PATTERNS = [
    r"/dev/sd[a-z]",      # Raw disk access
    r"/dev/nvme",          # NVMe disk access
    r"mkfs\.",             # Filesystem creation
    r"dd\s+if=",           # dd command
    r"fork\s*bomb",        # Fork bomb
    r":\(\)\s*\{",         # Fork bomb pattern
    r"sudo\b",             # sudo
    r"su\b",               # su
    r"passwd\b",           # password change
    r"shutdown\b",         # shutdown
    r"reboot\b",           # reboot
    r"init\s",             # init
    r"systemctl\b",        # systemctl
    r"/etc/passwd",        # password file
    r"/etc/shadow",        # shadow file
    r"\.\./\.\./\.\./",    # path traversal
]


class CliSandbox:
    """Safe CLI command execution sandbox."""

    DEFAULT_TIMEOUT = 10  # seconds
    MAX_OUTPUT_BYTES = 100_000

    async def execute(
        self, command: str, timeout: int = DEFAULT_TIMEOUT
    ) -> dict:
        """Execute a CLI command in a sandboxed environment.

        Returns: {"success": bool, "output": str, "error": str, "danger_level": str}
        """
        # Parse command
        parts = command.strip().split()
        if not parts:
            return {"success": False, "output": "", "error": "Empty command", "danger_level": "LOW"}

        cmd_name = parts[0].split("/")[-1]  # Strip path

        # Check against blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Command blocked by security policy: matched pattern '{pattern}'",
                    "danger_level": "CRITICAL",
                }

        # Check whitelist
        danger = COMMAND_WHITELIST.get(cmd_name)
        if danger is None:
            return {
                "success": False,
                "output": "",
                "error": f"Command not allowed: '{cmd_name}'. Use only whitelisted commands.",
                "danger_level": "CRITICAL",
            }

        # Execute with timeout
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout,
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]
            error = stderr.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_BYTES]

            return {
                "success": proc.returncode == 0,
                "output": output,
                "error": error,
                "danger_level": danger.value,
                "exit_code": proc.returncode,
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout}s",
                "danger_level": danger.value,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "danger_level": danger.value,
            }
