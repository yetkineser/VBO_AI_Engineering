"""
Main orchestrator: runs all fetchers, generates report, optionally schedules.
"""

import logging
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .arxiv_fetcher import fetch_arxiv_papers
from .github_fetcher import fetch_github_repos
from .rss_fetcher import fetch_rss_posts
from .hn_fetcher import fetch_hn_stories
from .report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def run_daily_digest(
    days_back_arxiv: int = 2,
    days_back_github: int = 7,
    days_back_rss: int = 2,
) -> tuple[Path, Path]:
    """Run all fetchers and generate a report."""

    console.print(Panel(
        "[bold cyan]Research Monitor[/bold cyan] — Daily Digest",
        subtitle=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    ))

    # arXiv
    console.print("\n[bold yellow]📄 Fetching arXiv papers...[/bold yellow]")
    papers = fetch_arxiv_papers(days_back=days_back_arxiv)
    console.print(f"   Found [green]{len(papers)}[/green] papers")

    # GitHub
    console.print("\n[bold yellow]🐙 Searching GitHub repos...[/bold yellow]")
    repos = fetch_github_repos(days_back=days_back_github)
    console.print(f"   Found [green]{len(repos)}[/green] repos")

    # RSS / Blogs
    console.print("\n[bold yellow]📝 Fetching blog posts (RSS)...[/bold yellow]")
    posts = fetch_rss_posts(days_back=days_back_rss)
    console.print(f"   Found [green]{len(posts)}[/green] relevant posts")

    # Hacker News
    console.print("\n[bold yellow]🟧 Checking Hacker News...[/bold yellow]")
    hn = fetch_hn_stories()
    console.print(f"   Found [green]{len(hn)}[/green] relevant stories")

    # Generate reports
    console.print("\n[bold yellow]📊 Generating reports...[/bold yellow]")
    md_path, html_path = generate_report(papers, repos, posts, hn)

    # Summary table
    summary = Table(title="Daily Digest Summary", show_header=True)
    summary.add_column("Source", style="cyan")
    summary.add_column("Count", style="green", justify="right")
    summary.add_row("arXiv Papers", str(len(papers)))
    summary.add_row("GitHub Repos", str(len(repos)))
    summary.add_row("Blog Posts", str(len(posts)))
    summary.add_row("Hacker News", str(len(hn)))
    summary.add_row("[bold]Total[/bold]", f"[bold]{len(papers) + len(repos) + len(posts) + len(hn)}[/bold]")
    console.print(summary)

    console.print(f"\n[green]✅ Reports saved:[/green]")
    console.print(f"   Markdown: {md_path}")
    console.print(f"   HTML:     {html_path}")

    _save_history(len(papers), len(repos), len(posts), len(hn))

    return md_path, html_path


def _save_history(papers: int, repos: int, posts: int, hn: int):
    """Append today's run to history.jsonl for tracking."""
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    history_file = data_dir / "history.jsonl"

    entry = {
        "date": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
        "repos": repos,
        "posts": posts,
        "hn_stories": hn,
        "total": papers + repos + posts + hn,
    }

    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Research Monitor - Daily AI/ML Digest")
    parser.add_argument(
        "--arxiv-days", type=int, default=2,
        help="How many days back to search arXiv (default: 2)",
    )
    parser.add_argument(
        "--github-days", type=int, default=7,
        help="How many days back to search GitHub (default: 7)",
    )
    parser.add_argument(
        "--rss-days", type=int, default=2,
        help="How many days back to search RSS feeds (default: 2)",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule (default: 08:00 UTC)",
    )
    parser.add_argument(
        "--schedule-time", type=str, default="08:00",
        help="Time to run daily digest (HH:MM UTC, default: 08:00)",
    )

    args = parser.parse_args()

    if args.schedule:
        import time
        import schedule as sched

        console.print(f"[bold]Scheduling daily digest at {args.schedule_time} UTC[/bold]")

        sched.every().day.at(args.schedule_time).do(
            run_daily_digest,
            days_back_arxiv=args.arxiv_days,
            days_back_github=args.github_days,
            days_back_rss=args.rss_days,
        )

        # Run once immediately
        run_daily_digest(
            days_back_arxiv=args.arxiv_days,
            days_back_github=args.github_days,
            days_back_rss=args.rss_days,
        )

        while True:
            sched.run_pending()
            time.sleep(60)
    else:
        run_daily_digest(
            days_back_arxiv=args.arxiv_days,
            days_back_github=args.github_days,
            days_back_rss=args.rss_days,
        )


if __name__ == "__main__":
    main()
