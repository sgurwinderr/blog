#!/usr/bin/env python3
"""
Generate complete course PDFs from HTML.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, CondPageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus.flowables import Flowable
from bs4 import BeautifulSoup
import re
import sys

class CodeBlock(Flowable):
    """Code block with background and border."""
    def __init__(self, code_text, style):
        Flowable.__init__(self)
        self.code_text = code_text
        self.style = style
        self.preformatted = Preformatted(code_text, style)

    def wrap(self, availWidth, availHeight):
        self.width, self.height = self.preformatted.wrap(availWidth - 20, availHeight)
        return self.width + 20, self.height + 20

    def draw(self):
        # Draw background
        self.canv.setFillColor(colors.HexColor('#f5f5f5'))
        self.canv.roundRect(0, 0, self.width + 20, self.height + 20, 4, fill=1, stroke=0)

        # Draw border
        self.canv.setStrokeColor(colors.HexColor('#cccccc'))
        self.canv.setLineWidth(1)
        self.canv.roundRect(0, 0, self.width + 20, self.height + 20, 4, fill=0, stroke=1)

        # Draw code text
        self.canv.saveState()
        self.canv.translate(10, 10)
        self.preformatted.drawOn(self.canv, 0, 0)
        self.canv.restoreState()

def parse_course_html(html_path):
    """Extract complete course content."""
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    modules = []
    module_sections = soup.find_all('section', class_='module')

    for section in module_sections:
        module_data = {}

        # Module header
        header = section.find('div', class_='module-header')
        if header:
            module_num = header.find('span', class_='module-number')
            module_title = header.find('h2')
            module_data['number'] = module_num.text.strip() if module_num else ""
            module_data['title'] = module_title.text.strip() if module_title else ""

        content = section.find('div', class_='module-content')
        if content:
            # The Problem
            problem_h3 = content.find('h3', string='The Problem')
            if problem_h3:
                problem_p = problem_h3.find_next('p')
                module_data['problem'] = problem_p.text.strip() if problem_p else ""

            # The Solution - extract PR and performance
            solution_h3 = content.find('h3', string='The Solution')
            if solution_h3:
                for elem in solution_h3.find_next_siblings():
                    if elem.name == 'h3':
                        break
                    if elem.name == 'p':
                        strong = elem.find('strong')
                        if strong:
                            if strong.text == 'PR:':
                                pr_text = elem.get_text()
                                module_data['pr_text'] = pr_text

                                link = elem.find('a')
                                if link:
                                    href = link.get('href', '')
                                    pr_match = re.search(r'/pull/(\d+)', href)
                                    if pr_match:
                                        module_data['pr_number'] = f"#{pr_match.group(1)}"
                                        module_data['pr_url'] = href
                            elif strong.text == 'Performance:':
                                module_data['performance'] = elem.get_text()

            # Code blocks
            code_blocks = []
            for pre in content.find_all('pre'):
                code = pre.find('code')
                if code:
                    code_blocks.append(code.get_text())
            module_data['code_blocks'] = code_blocks

            # The Pattern sections
            pattern_h3 = content.find('h3', string='The Pattern')
            if pattern_h3:
                pattern_sections = {}
                pattern_div = pattern_h3.find_next_sibling('div')
                if pattern_div:
                    subsections = pattern_div.find_all('div', recursive=False)
                    for div in subsections:
                        h4 = div.find('h4')
                        if h4:
                            section_title = h4.text.strip()
                            pattern_sections[section_title] = []
                            ul = div.find('ul')
                            if ul:
                                items = [li.text.strip() for li in ul.find_all('li')]
                                pattern_sections[section_title] = items
                module_data['pattern'] = pattern_sections

        modules.append(module_data)

    return modules

def create_course_pdf(html_path, pdf_path, course_title):
    """Generate PDF from course HTML."""
    modules = parse_course_html(html_path)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Styles
    module_number_style = ParagraphStyle(
        'ModuleNumber',
        fontName='Helvetica',
        fontSize=14,
        textColor=colors.grey,
        spaceAfter=4
    )

    module_title_style = ParagraphStyle(
        'ModuleTitle',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.black,
        spaceAfter=16
    )

    h3_style = ParagraphStyle(
        'H3',
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=8
    )

    h4_style = ParagraphStyle(
        'H4',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#444444'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.black,
        leading=14,
        spaceAfter=10
    )

    code_style = ParagraphStyle(
        'Code',
        fontName='Courier',
        fontSize=8,
        textColor=colors.black,
        leading=10
    )

    story = []

    for i, module in enumerate(modules):
        # Module number
        story.append(Paragraph(module.get('number', ''), module_number_style))

        # Title with PR
        module_title = module.get('title', '')
        pr_number = module.get('pr_number', '')
        pr_url = module.get('pr_url', '')

        if pr_number and pr_url:
            title_with_pr = f'{module_title} — PR <a href="{pr_url}" color="blue">{pr_number}</a>'
        elif pr_number:
            title_with_pr = f"{module_title} — PR {pr_number}"
        else:
            title_with_pr = module_title

        story.append(Paragraph(title_with_pr, module_title_style))
        story.append(Spacer(1, 0.2*inch))

        # The Problem
        if module.get('problem'):
            story.append(Paragraph("The Problem", h3_style))
            story.append(Paragraph(module['problem'], body_style))
            story.append(Spacer(1, 0.15*inch))

        # The Solution
        story.append(Paragraph("The Solution", h3_style))
        if module.get('pr_text'):
            story.append(Paragraph(module['pr_text'], body_style))
        if module.get('performance'):
            story.append(Paragraph(module['performance'], body_style))
        story.append(Spacer(1, 0.15*inch))

        # Code blocks
        if module.get('code_blocks'):
            for code in module['code_blocks']:
                story.append(CodeBlock(code, code_style))
                story.append(Spacer(1, 0.1*inch))

        # The Pattern - add conditional page break (only if not enough space)
        if module.get('pattern'):
            # CondPageBreak moves to new page if less than 3 inches available
            story.append(CondPageBreak(3*inch))
            story.append(Paragraph("The Pattern", h3_style))
            for section_name, items in module['pattern'].items():
                story.append(Paragraph(section_name, h4_style))
                for item in items:
                    story.append(Paragraph(f"• {item}", body_style))
            story.append(Spacer(1, 0.15*inch))

        # Page break after each module (ensures next PR starts on new page)
        if i < len(modules) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"✓ Created {pdf_path}")
    print(f"  {len(modules)} modules")

if __name__ == "__main__":
    # GPU Optimization Course
    create_course_pdf(
        '/home/gurwinde/blog/static/learn-ai/gpu-optimization-prs/index.html',
        'GPU-Optimization-Course.pdf',
        'GPU Optimization: 10 vLLM PRs'
    )

    # PyTorch Optimization Course
    create_course_pdf(
        '/home/gurwinde/blog/static/learn-ai/pytorch-optimization-prs/index.html',
        'PyTorch-Optimization-Course.pdf',
        'PyTorch Optimization: 10 PRs'
    )
