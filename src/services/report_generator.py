"""A PDF summary tied to the model and settings of one completed analysis."""
from datetime import datetime
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import config
from src.utils.privacy import redact_sensitive


def generate_pdf_report(record, output_path=None, chart_image_path=None):
    record = redact_sensitive(record)
    path = Path(output_path) if output_path else Path(config.REPORTS_DIR)/('report_' + str(record.get('id','analysis')) + '.pdf')
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    def para(value, style='BodyText'):
        return Paragraph(escape(str(value)), styles[style])
    story = [para('CrowdVision | Analysis report', 'Title'),
             para('Generated ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')), Spacer(1, .5*cm)]
    rows = [
        ['Input', record.get('file_name','Unknown')],
        ['Media', record.get('media_type','Unknown')],
        ['Actual model', record.get('model_name','Not recorded in this historical run')],
        ['Model classes', record.get('model_classes','Not recorded')],
        ['Peak visible count', record.get('max_head_count',0)],
        ['Average visible count', f"{record.get('avg_head_count',0):.2f}"],
        ['Final visible count', record.get('final_head_count',0)],
        ['Peak crowd level', record.get('peak_crowd_level','Not recorded')],
        ['Final crowd level', record.get('final_crowd_level',record.get('crowd_level','Not recorded'))],
        ['Analyzed samples', record.get('processed_frames',record.get('frames_processed','N/A'))],
        ['Processing time', f"{record.get('processing_time_seconds',0):.2f} s"],
    ]
    settings = record.get('settings',{})
    rows += [['Classification', settings.get('classification_mode','Not recorded')],
             ['Configured capacity', settings.get('max_capacity','Not recorded')],
             ['Source frame rate', record.get('source_fps','Not recorded')],
             ['Sample frame rate', record.get('sample_fps','Not recorded')]]
    table = Table([[para(a),para(b)] for a,b in rows], colWidths=[6*cm,10*cm])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf2f7')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),9),
                              ('TOPPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,-1),.3,colors.lightgrey)]))
    story += [table, Spacer(1,.5*cm)]
    if chart_image_path and Path(chart_image_path).is_file():
        story.append(Image(chart_image_path,width=16*cm,height=7*cm))
    story += [para('Interpretation', 'Heading2'),
              para('Peak and final crowd levels describe different points in the recording. Counts are visible detections, not unique people. Video is sampled; brief events between samples can be missed.'),
              Spacer(1,.25*cm),
              para('Occupancy levels use the configured capacity. Image coverage is an experimental box-area heuristic affected by perspective and zoom. Neither is a validated safety prediction. Accuracy must be evaluated for the target camera and scene.')]
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.5*cm,bottomMargin=1.5*cm).build(story)
    return str(path)
