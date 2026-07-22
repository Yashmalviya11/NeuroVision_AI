from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf(
    filename,
    patient_name,
    patient_id,
    age,
    gender,
    prediction,
    confidence
):

    styles = getSampleStyleSheet()

    pdf = SimpleDocTemplate(filename)

    story = []

    story.append(Paragraph("<b>NeuroVision AI Report</b>", styles["Title"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles["Normal"]))
    story.append(Paragraph(f"<b>Age:</b> {age}", styles["Normal"]))
    story.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["Heading2"]))

    story.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["Heading2"]
        )
    )

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(
        Paragraph(
            "This report was generated automatically by NeuroVision AI using Vision Transformer and Explainable AI.",
            styles["Normal"]
        )
    )

    pdf.build(story)