#!/usr/bin/env python3
"""Claude Code JSONL konusma gecmisini okunabilir Markdown'a cevirir."""

import json
import sys
import os
from datetime import datetime


def extract_text(content):
    """Message content'ten okunabilir metni cikarir."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", "").strip())
            elif btype == "tool_use":
                tool = block.get("name", "unknown")
                inp = block.get("input", {})
                if tool == "Read":
                    parts.append(f"*[Dosya okundu: `{inp.get('file_path', '?')}`]*")
                elif tool == "Write":
                    parts.append(f"*[Dosya yazildi: `{inp.get('file_path', '?')}`]*")
                elif tool == "Edit":
                    parts.append(f"*[Dosya duzenlendi: `{inp.get('file_path', '?')}`]*")
                elif tool == "Bash":
                    cmd = inp.get("command", "")
                    if len(cmd) > 120:
                        cmd = cmd[:120] + "..."
                    parts.append(f"*[Komut calistirildi: `{cmd}`]*")
                elif tool == "Grep":
                    parts.append(f"*[Arama: `{inp.get('pattern', '?')}`]*")
                elif tool == "Glob":
                    parts.append(f"*[Dosya arama: `{inp.get('pattern', '?')}`]*")
                elif tool == "Agent":
                    parts.append(f"*[Alt-ajan: {inp.get('description', '?')}]*")
                else:
                    parts.append(f"*[Arac: {tool}]*")
            elif btype == "tool_result":
                # Tool sonuclarini atla — cok uzun
                pass
            elif btype == "thinking":
                # Thinking bloklarini atla
                pass
        return "\n\n".join(p for p in parts if p)
    return ""


def convert_jsonl_to_md(jsonl_path, output_path):
    """JSONL dosyasini Markdown'a cevirir."""
    messages = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msg_type = obj.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue

            text = extract_text(obj.get("message", {}).get("content", ""))
            if not text:
                continue

            # Sadece tool_result olan user mesajlarini atla
            content = obj.get("message", {}).get("content", "")
            if isinstance(content, list):
                types = {b.get("type") for b in content}
                if types == {"tool_result"}:
                    continue

            messages.append({"role": msg_type, "text": text})

    # Markdown olustur
    lines = []
    lines.append("# Week 2: Metin Kodlama & Duygu Analizi — Konusma Gecmisi")
    lines.append("")
    lines.append(f"*Olusturulma tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    msg_num = 0
    for msg in messages:
        role = msg["role"]
        text = msg["text"]

        if role == "user":
            msg_num += 1
            lines.append(f"## {msg_num}. Kullanici")
            lines.append("")
            lines.append(text)
            lines.append("")
        else:
            lines.append("### Claude")
            lines.append("")
            lines.append(text)
            lines.append("")
        lines.append("---")
        lines.append("")

    md_content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Donusturuldu: {len(messages)} mesaj -> {output_path}")
    print(f"Dosya boyutu: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python jsonl_to_md.py <jsonl_dosyasi> [cikti.md]")
        print()
        print("Ornek:")
        print("  python jsonl_to_md.py ~/.claude/projects/.../konusma.jsonl outputs/konusma.md")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(jsonl_path)[0] + ".md"

    convert_jsonl_to_md(jsonl_path, output_path)
