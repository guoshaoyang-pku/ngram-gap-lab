#!/bin/bash
# sync_to_blog.sh — 把 ngram-gap-lab 的图与独立报告同步到 GitHub Pages 博客仓库
#
# ⚠️ 规则一：【绝不覆盖 blog 的 index.html】
#    blogs/ngram-gap-mechanism-guide/index.html 是手工维护的 9 章极简主线权威版
#    （见 agents.md §2）。主页面改动请直接编辑博客仓库那一份。
#
# ⚠️ 规则二：【不自动 push】
#    按 agents.md §0 P5，push 需用户确认。本脚本只做 copy + 可选 commit，
#    push 请手动执行，或用 blog-deploy skill。
#
# 用法：
#   bash docs/sync_to_blog.sh                          # 只复制，不 commit
#   bash docs/sync_to_blog.sh "sync(blog): 更新频率图"   # 复制 + commit（仍不 push）

set -uo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"                 # <repo>/docs
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
BLOG_DIR="${NGLAB_BLOG_ROOT:-$REPO_ROOT/../guoshaoyang-pku.github.io}"
TARGET_DIR="$BLOG_DIR/blogs/ngram-gap-mechanism-guide"

COMMIT_MSG="${1:-}"

if [ ! -d "$TARGET_DIR" ]; then
  echo "✗ 博客目标目录不存在: $TARGET_DIR" >&2
  exit 1
fi

copied=0
skipped=0

copy_one() {   # copy_one <src> <dst>
  if [ -e "$1" ]; then
    cp "$1" "$2" && echo "  ✓ $(basename "$2")" && copied=$((copied + 1))
  else
    echo "  – skip (missing): $1" && skipped=$((skipped + 1))
  fi
}

echo "=== 同步图与独立报告（不覆盖 index.html）==="

# 1. 主线图：docs/figs/main/*.svg + *.html → blog 根目录（index.html 用相对同级引用）
for f in "$SRC_DIR"/figs/main/*.svg "$SRC_DIR"/figs/main/*.html; do
  [ -e "$f" ] || continue
  copy_one "$f" "$TARGET_DIR/$(basename "$f")"
done

# 2. 理论 / toy 图：docs/figs/theory/ → blog figs/
if compgen -G "$SRC_DIR/figs/theory/*" > /dev/null 2>&1; then
  mkdir -p "$TARGET_DIR/figs"
  for f in "$SRC_DIR"/figs/theory/*.svg "$SRC_DIR"/figs/theory/*.png; do
    [ -e "$f" ] || continue
    copy_one "$f" "$TARGET_DIR/figs/$(basename "$f")"
  done
fi

# 3. 独立报告（本仓库生成后才会存在；index.html 通过链接而非 iframe 引用）
for name in exact-frequency-report.html sample285-mlp-comparison-report.html; do
  copy_one "$SRC_DIR/$name" "$TARGET_DIR/$name"
done

echo ""
echo "复制 $copied 个文件，跳过 $skipped 个。"

if [ -n "$COMMIT_MSG" ]; then
  echo ""
  echo "=== git commit（不 push）==="
  cd "$BLOG_DIR" || exit 1
  git add blogs/ngram-gap-mechanism-guide/
  git commit -m "$COMMIT_MSG" || echo "  (nothing to commit)"
  echo ""
  echo "已 commit。push 请手动执行：cd $BLOG_DIR && git push origin main"
else
  echo "未 commit（未提供 commit message）。"
fi
