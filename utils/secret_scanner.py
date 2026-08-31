"""
项目安全扫描工具 (Strict Secret Scanner)
功能：全量扫描项目中未被 .gitignore 忽略的代码文件（含注释），在严格模式下自动拦截以 'sk-' 开头及常见厂商格式的敏感 API Key。
支持独立运行或作为 pre-commit local hook 挂载。
"""

import fnmatch
import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple


class SecretScanner:
    """
    基于严格模式的代码机密泄漏检测器
    """

    # 默认强制忽略的系统/缓存目录
    DEFAULT_IGNORED_DIRS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".idea",
        ".vscode",
        ".gemini",
        "node_modules",
        "build",
        "dist",
        "test_samples_logs"
    }

    # 默认忽略的二进制与大型数据扩展名
    DEFAULT_IGNORED_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
        ".zip", ".tar", ".gz", ".7z",
        ".pyc", ".pyd", ".exe", ".dll", ".so",
        ".bin", ".pt", ".pth", ".safetensors", ".onnx",
        ".baseline"
    }

    # 严格模式检测正则库
    SECRET_PATTERNS = [
        # 1. 严格模式：捕获任何以 sk- 开头且紧随 6 位以上有效字符的字符串
        (
            "Generic/OpenAI/Qwen/DeepSeek API Key (sk-*)",
            re.compile(r"""\b(sk-[a-zA-Z0-9._\-]{6,})\b""")
        ),
        # 2. Google / Gemini API Key (以 AIza 开头)
        (
            "Google/Gemini API Key (AIza*)",
            re.compile(r"""\b(AIza[0-9A-Za-z\-_]{35})\b""")
        ),
        # 3. GitHub Personal Access Token (ghp_*, gho_*, ghu_*)
        (
            "GitHub Token (ghp/gho/ghu_*)",
            re.compile(r"""\b(gh[pousr]_[0-9a-zA-Z]{36})\b""")
        ),
        # 4. HuggingFace Token (hf_*)
        (
            "HuggingFace Token (hf_*)",
            re.compile(r"""\b(hf_[a-zA-Z0-9]{34,})\b""")
        ),
    ]

    # 行内白名单标记：如果该行包含此注释，则跳过拦截
    ALLOWLIST_COMMENT = "# pragma: allowlist secret"

    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.gitignore_patterns = self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> List[str]:
        """
        读取 .gitignore 中的所有过滤规则
        """
        patterns = []
        gitignore_path = os.path.join(self.root_dir, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # 处理目录通配符
                        if line.endswith("/"):
                            line = line[:-1]
                        patterns.append(line)
        return patterns

    def get_scannable_files(self) -> List[str]:
        """
        获取所有需要扫描的文件路径（严格遵循 .gitignore）
        优先使用 git ls-files 获取未被忽略的文件，保证与 git 行为 100% 一致
        """
        scannable = []
        try:
            # 使用 git 官方命令获取跟踪文件及未被忽略的未跟踪文件
            import subprocess
            out = subprocess.check_output(
                ["git", "ls-files", "-c", "-o", "--exclude-standard"],
                cwd=self.root_dir,
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                full = os.path.join(self.root_dir, line)
                if os.path.isfile(full) and not self.is_ignored_by_extension(line):
                    scannable.append(line)
            return scannable
        except Exception:
            pass

        # Fallback 遍历文件系统
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.DEFAULT_IGNORED_DIRS and not d.startswith(".")]
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self.root_dir).replace("\\", "/")
                if not self.is_ignored_by_extension(rel_path):
                    scannable.append(rel_path)
        return scannable

    def is_ignored_by_extension(self, file_rel_path: str) -> bool:
        """
        根据文件后缀或系统缓存目录进行轻量快速过滤
        """
        normalized = file_rel_path.replace("\\", "/")
        parts = normalized.split("/")
        for part in parts:
            if part in self.DEFAULT_IGNORED_DIRS:
                return True
        _, ext = os.path.splitext(normalized)
        return ext.lower() in self.DEFAULT_IGNORED_EXTENSIONS

    def scan_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        扫描单个文件中的机密
        """
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if self.ALLOWLIST_COMMENT in line:
                        continue

                    for rule_name, pattern in self.SECRET_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            raw_key = match.group(1).strip()
                            if raw_key.lower() in {"sk-your-key-here", "sk-xxx", "sk-placeholder"}:
                                continue

                            masked_key = (
                                raw_key[:7] + "****" + raw_key[-4:]
                                if len(raw_key) > 12
                                else raw_key[:4] + "****"
                            )

                            findings.append({
                                "line_number": line_idx,
                                "rule_name": rule_name,
                                "matched_snippet": line.strip(),
                                "masked_key": masked_key
                            })
                            break
        except Exception:
            pass

        return findings

    def run_scan(self) -> Tuple[int, Dict[str, List[Dict[str, Any]]]]:
        """
        执行全项目扫描
        返回: (scanned_file_count, findings_dict)
        """
        all_findings: Dict[str, List[Dict[str, Any]]] = {}
        files_to_scan = self.get_scannable_files()

        for rel_path in files_to_scan:
            full_path = os.path.join(self.root_dir, rel_path)
            if not os.path.exists(full_path):
                continue
            file_findings = self.scan_file(full_path)
            if file_findings:
                all_findings[rel_path] = file_findings

        return len(files_to_scan), all_findings


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    scanner = SecretScanner(root_dir=root_dir)

    print("=" * 65)
    print("[Strict Secret Scanner] 正在扫描全项目代码敏感 API Key...")
    print("模式: 【严格模式】(拦截所有以 'sk-' 开头及厂商密钥特征)")
    print("=" * 65)

    scanned_count, findings = scanner.run_scan()

    if not findings:
        print(f"\n[PASS] 安全检测通过！共扫描 {scanned_count} 个代码文件，未发现任何泄露的 API Key。")
        sys.exit(0)
    else:
        total_leaks = sum(len(items) for items in findings.values())
        print(f"\n[FAIL] 警告！检测到 {len(findings)} 个文件中存在敏感密钥硬编码 (共 {total_leaks} 处泄露)：\n")

        for file_path, items in findings.items():
            print(f">> 文件: {file_path}")
            for item in items:
                print(f"  |-- [第 {item['line_number']} 行] 特征: {item['rule_name']}")
                print(f"  |-- 检测命中: {item['masked_key']}")
                print(f"  +-- 代码预览: {item['matched_snippet']}")
                print()

        print("-" * 65)
        print("建议处置措施:")
        print("1. 请立即将上述真实 Key 移出代码，统一存放在 .env 文件中并通过 os.getenv() 读取；")
        print("2. 若某行为有意保留的公开 Mock 示例，可在该行末尾添加注释: # pragma: allowlist secret 进行豁免。")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
