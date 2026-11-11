"""Run with bundled Python (reportlab, pdfplumber, PIL installed)."""
from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pdfplumber

ROOT=Path(__file__).resolve().parent
fonts=Path(r'C:\Users\RandomX77\AppData\Roaming\Python\Python314\site-packages\matplotlib\mpl-data\fonts\ttf')
for name,file in [('DV','DejaVuSans.ttf'),('DV-Bold','DejaVuSans-Bold.ttf'),('DV-Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyRU',fontName='DV',fontSize=8.8,leading=12,spaceAfter=5.5))
styles.add(ParagraphStyle(name='TitleRU',fontName='DV-Bold',fontSize=20,leading=25,spaceAfter=18,textColor=colors.HexColor('#12364a')))
styles.add(ParagraphStyle(name='H1RU',fontName='DV-Bold',fontSize=13,leading=17,spaceBefore=14,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#12364a')))
styles.add(ParagraphStyle(name='H2RU',fontName='DV-Bold',fontSize=10.5,leading=14,spaceBefore=10,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='CellRU',fontName='DV',fontSize=7,leading=9,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CodeRU',fontName='DV-Mono',fontSize=7.2,leading=10,spaceAfter=8))

def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\((https?[^)]+)\)',r'<link href="\2" color="#175c81">\1</link>',s)
    s=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="DV-Mono" size="8">\1</font>',s)
    return s

def render(path):
    lines=path.read_text(encoding='utf8').splitlines();story=[];i=0;width=A4[0]-84
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            i+=1;code=[]
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(code),styles['CodeRU'],maxLineLength=95));i+=1;continue
        if line.startswith('|'):
            raw=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                vals=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',v) for v in vals):raw.append(vals)
                i+=1
            n=len(raw[0]);tw=[width/n]*n
            data=[[Paragraph(inline(v),styles['CellRU']) for v in r] for r in raw]
            t=Table(data,colWidths=tw,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8ed')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#bdcdd5')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([t,Spacer(1,9)]);continue
        match=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            from PIL import Image as PI
            p=ROOT/match[2];im=PI.open(p);h=width*im.height/im.width
            story.extend([Image(str(p),width=width,height=h),Spacer(1,8)]);i+=1;continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));style='TitleRU' if level==1 else ('H1RU' if level==2 else 'H2RU')
            story.append(Paragraph(inline(line.lstrip('#').strip()),styles[style]));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','```','![')):
            para.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(para)),styles['BodyRU']))
    def footer(canvas,doc):
        canvas.setFont('DV',7);canvas.setFillColor(colors.HexColor('#557080'));canvas.drawString(42,23,'PQT–QCat LIFE-001 · continuous plasticity · 2026-11-11');canvas.drawRightString(A4[0]-42,23,str(doc.page))
    output=path.with_suffix('.pdf')
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=32,bottomMargin=34,title='PQT–QCat LIFE-001: continuous plasticity',author='Independent computational research').build(story,onFirstPage=footer,onLaterPages=footer)
    with pdfplumber.open(output) as pdf:
        print('PDF',len(pdf.pages),'pages')
        for idx in [0,2,len(pdf.pages)-1]:pdf.pages[idx].to_image(resolution=100).save(ROOT/f'report_preview_{idx+1}.png')
        assert all((p.extract_text() or '').strip() for p in pdf.pages)

if __name__=='__main__':render(ROOT/'LIFE_001_REPORT_RU.md')
