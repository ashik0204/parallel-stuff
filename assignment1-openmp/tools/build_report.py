# Builds report.md / report.html / report.pdf from report_template.md and results/.
# Usage (from the cs433 folder):  python3 tools/build_report.py
import glob, os, subprocess, sys

sys.path.insert(0, 'tools/pylib')                     # optional local copy of 'markdown'
subprocess.run([sys.executable, 'tools/make_tables.py'], check=True, stdout=subprocess.DEVNULL)

t = open('report_template.md').read()
tab = open('results/tables.md').read()
q1, q2 = tab.split('#### n = 1,000', 1)
q2 = '#### n = 1,000' + q2
t = t.replace('{{Q1_TABLES}}', "Times are in seconds (median of 3 runs), computation only: allocation and array "
              "initialization are excluded. *Build H* = triangle listing and conflict graph construction; "
              "*Coloring* = batched DSATUR; *Improve* = lower bound and top-class elimination.\n" + q1)
t = t.replace('{{Q2_TABLES}}', "Times are in milliseconds (median of 5 runs), measured on instances made by "
              "`InitializeInput`. *GB/s* = bytes of A (non-zeros) divided by the solve time of the pipelined version.\n" + q2)
t = t.replace('{{Q1_DISCUSSION}}', open('results/q1_discussion.md').read())
t = t.replace('{{Q2_DISCUSSION}}', open('results/q2_discussion.md').read())
assert '{{' not in t
open('report.md', 'w').write(t)
print('wrote report.md')

try:
    import markdown
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', 'tools/pylib', 'markdown'], check=True)
    import markdown
body = markdown.markdown(t, extensions=['tables'])
css = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 19pt; margin: 0 0 4pt; } h2 { font-size: 14pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt; }
h3 { font-size: 12pt; margin-top: 13pt; } h4 { font-size: 10.5pt; margin: 12pt 0 3pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 5pt 0 9pt; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2.5pt 7pt; } th { background: #eee; }
td { text-align: right; } td:first-child { text-align: left; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
hr { border: 0; border-top: 1px solid #ccc; margin: 16pt 0; }
li { margin: 2pt 0; }
"""
open('report.html', 'w').write("<!doctype html><html><head><meta charset='utf-8'><title>CS433 Assignment 1 Report</title>"
                               f"<style>{css}</style></head><body>{body}</body></html>")
print('wrote report.html')
for chrome in ['google-chrome', 'chromium-browser', 'chromium']:
    try:
        subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                        '--print-to-pdf=report.pdf', 'report.html'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('wrote report.pdf')
        break
    except (FileNotFoundError, subprocess.CalledProcessError):
        continue
else:
    print('no Chrome found: open report.html in a browser and print it to PDF')
