#!/usr/bin/env python3
"""Convert a benchmark report Markdown file into a self-contained HTML document."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
from pathlib import Path
import re


def _slugify(text: str) -> str:
  plain = re.sub(r"`([^`]+)`", r"\1", text)
  plain = re.sub(r"[^\w\s-]", "", plain.lower())
  return re.sub(r"[\s_]+", "-", plain.strip())


def _render_inline(text: str, base_dir: Path, embed_images: bool = True) -> str:
  """Render inline Markdown (images, code spans, links, bold, italic)."""
  tokens: dict[str, str] = {}

  def _stash(html_fragment: str) -> str:
    key = f"\x00TOK{len(tokens)}\x00"
    tokens[key] = html_fragment
    return key

  # 1. Inline code spans `...`
  def _repl_code(m: re.Match[str]) -> str:
    return _stash(f"<code>{html.escape(m.group(1))}</code>")

  text = re.sub(r"`([^`]+)`", _repl_code, text)

  # 2. Images ![alt](src)
  def _repl_img(m: re.Match[str]) -> str:
    alt = html.escape(m.group(1))
    src = m.group(2).strip()
    img_path = (base_dir / src).resolve()
    if embed_images and not src.startswith(("http://", "https://", "data:")) and img_path.is_file():
      mime, _ = mimetypes.guess_type(str(img_path))
      mime = mime or "image/png"
      b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
      src_attr = f"data:{mime};base64,{b64}"
    else:
      src_attr = html.escape(src)
    return _stash(
        f'<figure class="report-figure">'
        f'<img src="{src_attr}" alt="{alt}" loading="lazy" />'
        f'<figcaption>{alt}</figcaption>'
        f'</figure>'
    )

  text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl_img, text)

  # 3. Links [label](href)
  def _repl_link(m: re.Match[str]) -> str:
    label_raw = m.group(1)
    href = html.escape(m.group(2).strip())
    # Render inline formatting inside link label
    label_html = _render_inline_text_only(label_raw, tokens)
    return _stash(f'<a href="{href}">{label_html}</a>')

  text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl_link, text)

  text = _render_inline_text_only(text, tokens)
  for k, v in tokens.items():
    text = text.replace(k, v)
  return text


def _render_inline_text_only(text: str, tokens: dict[str, str]) -> str:
  escaped = html.escape(text)
  # Restore any stashed tokens whose keys had no special HTML chars
  escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
  escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
  escaped = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", escaped)
  for k, v in tokens.items():
    escaped = escaped.replace(k, v)
  return escaped


def _parse_list(lines: list[str], start: int, base_dir: Path, embed_images: bool) -> tuple[str, int]:
  """Parse nested unordered/ordered Markdown lists starting at `start`."""
  list_item_re = re.compile(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$")

  def _collect_items(idx: int, base_indent: int, ordered: bool) -> tuple[str, int]:
    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    n = len(lines)
    while idx < n:
      line = lines[idx]
      if not line.strip():
        # Check if list continues after blank line at >= base_indent
        look = idx + 1
        while look < n and not lines[look].strip():
          look += 1
        if look < n:
          m_next = list_item_re.match(lines[look])
          if m_next and len(m_next.group(1)) >= base_indent:
            idx = look
            continue
        break

      m = list_item_re.match(line)
      if not m:
        break
      indent = len(m.group(1))
      is_ord = m.group(3) is not None
      if indent < base_indent:
        break
      if indent > base_indent:
        sub_html, idx = _collect_items(idx, indent, is_ord)
        if out[-1].endswith("</li>"):
          out[-1] = out[-1][:-5] + "\n" + sub_html + "</li>"
        else:
          out.append(sub_html)
        continue
      if is_ord != ordered:
        break

      item_lines = [m.group(4)]
      idx += 1
      sub_chunks: list[str] = []
      while idx < n:
        nxt = lines[idx]
        if not nxt.strip():
          break
        m_sub = list_item_re.match(nxt)
        if m_sub:
          sub_indent = len(m_sub.group(1))
          if sub_indent > base_indent:
            sub_html, idx = _collect_items(idx, sub_indent, m_sub.group(3) is not None)
            sub_chunks.append(sub_html)
            continue
          break
        nxt_indent = len(nxt) - len(nxt.lstrip(" "))
        if nxt_indent > base_indent:
          item_lines.append(nxt.strip())
          idx += 1
        else:
          break

      item_html = _render_inline(" ".join(item_lines), base_dir, embed_images)
      if sub_chunks:
        out.append(f"<li>{item_html}\n" + "\n".join(sub_chunks) + "</li>")
      else:
        out.append(f"<li>{item_html}</li>")

    out.append(f"</{tag}>")
    return "\n".join(out), idx

  m0 = list_item_re.match(lines[start])
  assert m0 is not None
  return _collect_items(start, len(m0.group(1)), m0.group(3) is not None)


def _split_table_row(line: str) -> list[str]:
  s = line.strip()
  if s.startswith("|"):
    s = s[1:]
  if s.endswith("|"):
    s = s[:-1]
  return [cell.strip() for cell in s.split("|")]


def markdown_to_html(md_text: str, base_dir: Path, embed_images: bool = True) -> str:
  lines = md_text.splitlines()
  n = len(lines)
  i = 0
  body_parts: list[str] = []
  title = "AI Edge Bench Report"
  list_item_re = re.compile(r"^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$")

  while i < n:
    line = lines[i]
    stripped = line.strip()

    if not stripped:
      i += 1
      continue

    # Fenced code block
    if stripped.startswith("```"):
      lang = stripped[3:].strip()
      code_lines: list[str] = []
      i += 1
      while i < n and not lines[i].strip().startswith("```"):
        code_lines.append(lines[i])
        i += 1
      if i < n:
        i += 1
      cls = f' class="language-{html.escape(lang)}"' if lang else ""
      body_parts.append(f"<pre><code{cls}>{html.escape(chr(10).join(code_lines))}</code></pre>")
      continue

    # Heading
    m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
    if m_head:
      level = len(m_head.group(1))
      raw_heading = m_head.group(2).strip()
      if level == 1 and title == "AI Edge Bench Report":
        title = re.sub(r"`([^`]+)`", r"\1", raw_heading)
      slug = _slugify(raw_heading)
      heading_html = _render_inline(raw_heading, base_dir, embed_images)
      body_parts.append(f'<h{level} id="{slug}">{heading_html}</h{level}>')
      i += 1
      continue

    # Blockquote
    if stripped.startswith(">"):
      bq_lines: list[str] = []
      while i < n and lines[i].strip().startswith(">"):
        bq_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
        i += 1
      bq_html = _render_inline(" ".join(bq_lines), base_dir, embed_images)
      body_parts.append(f"<blockquote><p>{bq_html}</p></blockquote>")
      continue

    # Table
    if "|" in stripped and i + 1 < n and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i + 1].strip()):
      headers = _split_table_row(lines[i])
      i += 2
      rows: list[list[str]] = []
      while i < n and "|" in lines[i] and lines[i].strip():
        rows.append(_split_table_row(lines[i]))
        i += 1
      th_html = "".join(f"<th>{_render_inline(h, base_dir, embed_images)}</th>" for h in headers)
      tr_list: list[str] = []
      for r in rows:
        cells = []
        for idx_c, c in enumerate(r):
          rendered = _render_inline(c, base_dir, embed_images)
          cells.append(f"<td>{rendered}</td>")
        tr_list.append("<tr>" + "".join(cells) + "</tr>")
      body_parts.append(
          '<div class="table-wrap"><table>'
          f"<thead><tr>{th_html}</tr></thead>"
          f"<tbody>{''.join(tr_list)}</tbody>"
          "</table></div>"
      )
      continue

    # List
    if list_item_re.match(line):
      list_html, i = _parse_list(lines, i, base_dir, embed_images)
      body_parts.append(list_html)
      continue

    # Standalone image paragraph
    if re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped):
      body_parts.append(_render_inline(stripped, base_dir, embed_images))
      i += 1
      continue

    # Paragraph
    para_lines: list[str] = []
    while i < n:
      cur = lines[i].strip()
      if (
          not cur
          or cur.startswith("#")
          or cur.startswith("```")
          or cur.startswith(">")
          or list_item_re.match(lines[i])
          or ("|" in cur and i + 1 < n and re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i + 1].strip()))
      ):
        break
      para_lines.append(cur)
      i += 1
    para_html = _render_inline(" ".join(para_lines), base_dir, embed_images)
    body_parts.append(f"<p>{para_html}</p>")

  body_joined = "\n".join(body_parts)
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #1f2328;
      --muted: #59636e;
      --border: #d1d9e0;
      --surface: #f6f8fa;
      --accent: #0969da;
      --code-bg: #eff1f3;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem 1.5rem 4rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      font-size: 15px;
      line-height: 1.6;
      color: var(--fg);
      background: var(--bg);
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
    }}
    h1, h2, h3, h4 {{
      margin-top: 1.8rem;
      margin-bottom: 0.75rem;
      font-weight: 600;
      line-height: 1.25;
    }}
    h1 {{
      font-size: 1.85rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border);
    }}
    h2 {{
      font-size: 1.4rem;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid var(--border);
    }}
    h3 {{
      font-size: 1.15rem;
    }}
    p, ul, ol, blockquote {{
      margin-top: 0;
      margin-bottom: 1rem;
    }}
    ul, ol {{
      padding-left: 1.75rem;
    }}
    li {{
      margin-bottom: 0.25rem;
    }}
    li > ul, li > ol {{
      margin-top: 0.25rem;
      margin-bottom: 0.25rem;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    blockquote {{
      margin-left: 0;
      padding: 0.5rem 1rem;
      color: var(--muted);
      border-left: 4px solid var(--border);
      background: var(--surface);
      border-radius: 0 6px 6px 0;
    }}
    blockquote p:last-child {{
      margin-bottom: 0;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.88em;
      padding: 0.15em 0.35em;
      background: var(--code-bg);
      border-radius: 4px;
    }}
    pre {{
      padding: 1rem;
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    pre code {{
      padding: 0;
      background: transparent;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 1rem 0 1.5rem;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      white-space: nowrap;
    }}
    th, td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid var(--border);
      border-right: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th:last-child, td:last-child {{
      border-right: none;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    thead th {{
      background: var(--surface);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tbody tr:nth-child(even) {{
      background: #fafbfc;
    }}
    tbody tr:hover {{
      background: #f0f4f8;
    }}
    .report-figure {{
      margin: 1.5rem 0;
      text-align: center;
    }}
    .report-figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    .report-figure figcaption {{
      margin-top: 0.4rem;
      font-size: 0.85rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
{body_joined}
  </main>
</body>
</html>
"""


def main() -> int:
  parser = argparse.ArgumentParser(description="Convert Markdown benchmark report to HTML.")
  parser.add_argument("input", type=Path, help="Path to input README.md")
  parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path (default: README.html next to input)")
  parser.add_argument("--no-embed-images", action="store_true", help="Link images by relative path instead of base64 data URIs")
  args = parser.parse_args()

  md_path = args.input.resolve()
  out_path = (args.output or md_path.with_suffix(".html")).resolve()
  html_doc = markdown_to_html(
      md_path.read_text(encoding="utf-8"),
      base_dir=md_path.parent,
      embed_images=not args.no_embed_images,
  )
  out_path.write_text(html_doc, encoding="utf-8")
  print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
