import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
import torch

from src.inference import MoodInference
from src.features import extract_math_features_from_array
import io
import soundfile as sf
from datetime import datetime

# PDF generation imports (optional)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

def create_audio_slice_bytes(y_slice, sr):
    """
    Create a temporary WAV file in memory for an audio slice.
    
    Args:
        y_slice: Audio slice array
        sr: Sample rate
    
    Returns:
        BytesIO object containing WAV data
    """
    # Create BytesIO buffer
    audio_buffer = io.BytesIO()
    
    # Write audio to buffer as WAV
    sf.write(audio_buffer, y_slice, sr, format='WAV')
    
    # Reset buffer position to beginning
    audio_buffer.seek(0)
    
    return audio_buffer

def generate_pdf_report(data):
    """
    Generate a comprehensive PDF report of the audio mood analysis including all visualizations.
    
    Args:
        data: Dictionary containing analysis results, plots, and visualizations
    
    Returns:
        BytesIO buffer containing PDF data, or None if generation fails
    """
    if not PDF_AVAILABLE:
        return None
        
    try:
        # Create PDF buffer
        pdf_buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor='#1f4e79',  # Dark blue
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#2d5016',  # Dark green
            spaceAfter=12
        )
        
        # Build story (content)
        story = []
        
        # Title
        story.append(Paragraph("🎵 AI Music Mood Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", styles['Normal']))
        story.append(Paragraph(f"<b>Audio File:</b> {data.get('filename', 'Unknown')}", styles['Normal']))
        story.append(Paragraph(f"<b>Duration:</b> {data.get('duration', 0):.1f} seconds", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Main Prediction
        story.append(Paragraph("🎭 Main Prediction", heading_style))
        final_mood = data.get('final_mood', 'Unknown')
        story.append(Paragraph(f"<b>Predicted Mood:</b> {final_mood.title()}", styles['Normal']))
        
        # Confidence scores
        final_conf = data.get('final_confidence', [0, 0, 0, 0])
        labels = ['Happy', 'Sad', 'Calm', 'Energetic']
        
        conf_data = [['Mood', 'Confidence Score']]
        for label, conf in zip(labels, final_conf):
            conf_data.append([label, f"{conf*100:.1f}%"])
        
        conf_table = Table(conf_data)
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#808080'),  # Grey
            ('TEXTCOLOR', (0, 0), (-1, 0), '#f5f5f5'),   # Whitesmoke
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#f5f5dc'),  # Beige
            ('GRID', (0, 0), (-1, -1), 1, '#000000')     # Black
        ]))
        
        story.append(conf_table)
        story.append(Spacer(1, 20))
        
        # Page break before visualizations
        from reportlab.platypus import PageBreak
        story.append(PageBreak())
        
        # VISUALIZATIONS SECTION
        story.append(Paragraph("📊 AUDIO ANALYSIS VISUALIZATIONS", title_style))
        story.append(Spacer(1, 20))
        
        # Add Waveform Visualization
        if 'waveform_img' in data and data['waveform_img']:
            story.append(Paragraph("1. Audio Waveform Analysis", heading_style))
            story.append(Paragraph("Visual representation of the audio signal amplitude over time showing energy distribution:", styles['Normal']))
            story.append(Spacer(1, 8))
            
            waveform_img = Image(data['waveform_img'], width=6.5*inch, height=2.5*inch)
            story.append(waveform_img)
            story.append(Spacer(1, 20))
        
        # Add Mel Spectrogram
        if 'spectrogram_img' in data and data['spectrogram_img']:
            story.append(Paragraph("2. Mel-Spectrogram Analysis", heading_style))
            story.append(Paragraph("Frequency content analysis showing how different frequencies contribute to mood detection:", styles['Normal']))
            story.append(Spacer(1, 8))
            
            spec_img = Image(data['spectrogram_img'], width=6.5*inch, height=2.5*inch)
            story.append(spec_img)
            story.append(Spacer(1, 20))
        
        # Add Radar Chart and Timeline side by side
        if ('radar_img' in data and data['radar_img']) or ('heatmap_img' in data and data['heatmap_img']):
            story.append(Paragraph("3. Confidence Analysis & Timeline", heading_style))
            
            # Create a table to place radar and timeline side by side
            viz_data = []
            viz_row = []
            
            if 'radar_img' in data and data['radar_img']:
                radar_img = Image(data['radar_img'], width=3*inch, height=3*inch)
                viz_row.append(radar_img)
            else:
                viz_row.append("")
                
            if 'heatmap_img' in data and data['heatmap_img']:
                heatmap_img = Image(data['heatmap_img'], width=3.5*inch, height=2*inch)
                viz_row.append(heatmap_img)
            else:
                viz_row.append("")
            
            if viz_row:
                viz_data.append(viz_row)
                viz_table = Table(viz_data, colWidths=[3.2*inch, 3.8*inch])
                viz_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(viz_table)
                story.append(Spacer(1, 10))
                
            story.append(Paragraph("<b>Left:</b> Multi-dimensional confidence radar showing prediction strength across all mood categories", styles['Normal']))
            story.append(Paragraph("<b>Right:</b> Timeline heatmap showing mood evolution throughout the audio", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Page break before detailed analysis
        story.append(PageBreak())
        
        # DETAILED ANALYSIS SECTION
        story.append(Paragraph("📈 DETAILED ANALYSIS RESULTS", title_style))
        story.append(Spacer(1, 20))
        
        # Audio Statistics Summary
        story.append(Paragraph("1. Audio Statistics Summary", heading_style))
        
        audio_stats_data = [
            ['Property', 'Value', 'Interpretation'],
            ['File Duration', f"{data.get('duration', 0):.2f} seconds", 'Total audio length'],
            ['Sample Rate', '22050 Hz', 'Audio quality/resolution'],
            ['Total Segments', f"{len(data.get('slice_moods', []))}", 'Number of analyzed parts'],
            ['Analysis Window', f"{data.get('window_size', 5.0)} seconds", 'Size of each segment'],
            ['Overlap', '2.5 seconds', 'Segment overlap for smooth analysis']
        ]
        
        stats_table = Table(audio_stats_data, colWidths=[2*inch, 2*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#2E8B57'),  # Sea green header
            ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#F0F8FF'),  # Alice blue
            ('GRID', (0, 0), (-1, -1), 1, '#000000'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Mathematical Audio Features
        if 'math_features' in data and data['math_features']:
            story.append(Paragraph("2. Mathematical Audio Features", heading_style))
            story.append(Paragraph("Advanced mathematical properties extracted from the audio signal for AI analysis:", styles['Normal']))
            story.append(Spacer(1, 10))
            
            math_features = data['math_features']
            
            # Create summary table of average features
            if len(math_features) > 0:
                avg_rms = np.mean([f['rms'] for f in math_features])
                avg_zcr = np.mean([f['zcr'] for f in math_features])
                avg_centroid = np.mean([f['centroid'] for f in math_features])
                avg_rolloff = np.mean([f['rolloff'] for f in math_features])
                avg_entropy = np.mean([f['entropy'] for f in math_features])
                avg_pitch_var = np.mean([f['pitch_var'] for f in math_features])
                
                math_data = [
                    ['Feature', 'Average Value', 'Description', 'Mood Relevance'],
                    ['RMS Energy', f"{avg_rms:.4f}", 'Overall loudness/energy level', 'High energy → Energetic moods'],
                    ['Zero Crossing Rate', f"{avg_zcr:.4f}", 'Signal crossing zero amplitude', 'High ZCR → Noisy/Energetic'],
                    ['Spectral Centroid', f"{avg_centroid:.0f} Hz", 'Brightness of sound', 'High centroid → Happy/Bright'],
                    ['Spectral Rolloff', f"{avg_rolloff:.0f} Hz", 'High frequency content', 'High rolloff → Complex/Rich'],
                    ['Spectral Entropy', f"{avg_entropy:.3f}", 'Frequency distribution complexity', 'High entropy → Complex moods'],
                    ['Pitch Variance', f"{avg_pitch_var:.3f}", 'Pitch stability measure', 'High variance → Dynamic moods']
                ]
                
                math_table = Table(math_data, colWidths=[1.3*inch, 1.2*inch, 1.8*inch, 2.2*inch])
                math_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),  # Blue header
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),   # White text
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#F2F2F2'),  # Light grey
                    ('GRID', (0, 0), (-1, -1), 1, '#000000'),     # Black
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                
                story.append(math_table)
                story.append(Spacer(1, 20))
        
        # Detailed Confidence Analysis
        story.append(Paragraph("3. Detailed Confidence Analysis", heading_style))
        
        final_conf = data.get('final_confidence', [0, 0, 0, 0])
        labels = ['Happy', 'Sad', 'Calm', 'Energetic']
        
        # Confidence interpretation
        confidence_interpretations = []
        for i, (label, conf) in enumerate(zip(labels, final_conf)):
            if conf > 0.7:
                interpretation = "Very High - Strong indication"
            elif conf > 0.5:
                interpretation = "High - Moderate indication"
            elif conf > 0.3:
                interpretation = "Medium - Some indication"
            else:
                interpretation = "Low - Weak indication"
            confidence_interpretations.append([label, f"{conf*100:.1f}%", interpretation])
        
        conf_detail_data = [['Mood Category', 'Confidence Score', 'Interpretation']]
        conf_detail_data.extend(confidence_interpretations)
        
        conf_detail_table = Table(conf_detail_data, colWidths=[2*inch, 2*inch, 2.5*inch])
        conf_detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#8B4513'),  # Saddle brown
            ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#FFF8DC'),  # Cornsilk
            ('GRID', (0, 0), (-1, -1), 1, '#000000'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(conf_detail_table)
        story.append(Spacer(1, 20))
        
        # Page break before segment analysis
        story.append(PageBreak())
        
        # SEGMENT-BY-SEGMENT ANALYSIS
        story.append(Paragraph("🔍 SEGMENT-BY-SEGMENT ANALYSIS", title_style))
        story.append(Spacer(1, 20))
        
        slice_moods = data.get('slice_moods', [])
        slice_confs = data.get('slice_confidences', [])
        offsets = data.get('offsets', [])
        window_size = data.get('window_size', 5.0)
        
        if slice_moods:
            # Analysis Overview
            story.append(Paragraph("4. Analysis Overview", heading_style))
            
            most_common = max(set(slice_moods), key=slice_moods.count) if slice_moods else 'Unknown'
            avg_conf = np.mean([np.max(conf) for conf in slice_confs]) if slice_confs else 0
            
            overview_data = [
                ['Metric', 'Value', 'Details'],
                ['Total Segments', f"{len(slice_moods)}", f"Audio split into {window_size}s segments"],
                ['Dominant Mood', most_common.title(), f"Most frequent prediction across segments"],
                ['Average Confidence', f"{avg_conf*100:.1f}%", f"Mean confidence across all predictions"],
                ['Analysis Method', 'Neural Network', 'Deep learning model with mel-spectrogram input'],
                ['Feature Extraction', 'Librosa + Custom', 'Mathematical and spectral feature analysis']
            ]
            
            overview_table = Table(overview_data, colWidths=[2*inch, 1.8*inch, 2.7*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#800080'),  # Purple header
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), '#E6E6FA'),  # Lavender
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            story.append(overview_table)
            story.append(Spacer(1, 20))
            
            # Mood Transitions Analysis
            story.append(Paragraph("5. Mood Transitions Analysis", heading_style))
            story.append(Paragraph("Analysis of how moods change throughout the audio timeline:", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Calculate transitions
            transitions = []
            for i in range(len(slice_moods) - 1):
                current_mood = slice_moods[i]
                next_mood = slice_moods[i + 1]
                current_time = offsets[i] if i < len(offsets) else i * window_size
                next_time = offsets[i + 1] if (i + 1) < len(offsets) else (i + 1) * window_size
                
                if current_mood != next_mood:
                    transitions.append({
                        'from_mood': current_mood.title(),
                        'to_mood': next_mood.title(),
                        'time': f"{current_time:.1f}s → {next_time:.1f}s",
                        'type': 'Mood Change'
                    })
            
            if transitions:
                transition_data = [['Transition', 'From Mood', 'To Mood', 'Time Range']]
                for i, trans in enumerate(transitions[:8]):  # Show first 8 transitions
                    transition_data.append([
                        f"#{i+1}",
                        trans['from_mood'],
                        trans['to_mood'],
                        trans['time']
                    ])
                
                if len(transitions) > 8:
                    transition_data.append(['...', '...', '...', f'({len(transitions)-8} more transitions)'])
                
                transition_table = Table(transition_data, colWidths=[1*inch, 1.8*inch, 1.8*inch, 2*inch])
                transition_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), '#FF6347'),  # Tomato header
                    ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), '#FFE4E1'),  # Misty rose
                    ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                
                story.append(transition_table)
            else:
                story.append(Paragraph("<b>No significant mood transitions detected.</b> The audio maintains a consistent mood throughout.", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Detailed Segment Analysis
            story.append(Paragraph("6. Detailed Segment Analysis", heading_style))
            story.append(Paragraph("Individual segment predictions with confidence breakdown:", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Detailed slice table (limit to first 12 for readability)
            slice_data = [['#', 'Time Range', 'Mood', 'Confidence', 'Happy%', 'Sad%', 'Calm%', 'Energetic%']]
            max_slices_in_pdf = min(12, len(slice_moods))
            
            for i in range(max_slices_in_pdf):
                mood = slice_moods[i]
                offset = offsets[i] if i < len(offsets) else i * window_size
                conf = slice_confs[i] if i < len(slice_confs) else [0, 0, 0, 0]
                max_conf = np.max(conf) if len(conf) > 0 else 0
                
                slice_data.append([
                    str(i + 1),
                    f"{offset:.1f}-{offset + window_size:.1f}s",
                    mood.title(),
                    f"{max_conf*100:.1f}%",
                    f"{conf[0]*100:.0f}%" if len(conf) > 0 else "0%",
                    f"{conf[1]*100:.0f}%" if len(conf) > 1 else "0%",
                    f"{conf[2]*100:.0f}%" if len(conf) > 2 else "0%",
                    f"{conf[3]*100:.0f}%" if len(conf) > 3 else "0%"
                ])
            
            if len(slice_moods) > max_slices_in_pdf:
                slice_data.append(['...', '...', '...', '...', '...', '...', '...', f'({len(slice_moods) - max_slices_in_pdf} more)'])
            
            slice_table = Table(slice_data, colWidths=[0.4*inch, 1.2*inch, 1*inch, 0.8*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.8*inch])
            slice_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#20B2AA'),  # Light sea green
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), '#F0FFFF'),  # Azure
                ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            
            story.append(slice_table)
        
        story.append(Spacer(1, 20))
        
        # Analysis Features
        story.append(Paragraph("📊 Analysis Features", heading_style))
        story.append(Paragraph(
            "This analysis used advanced AI techniques including:",
            styles['Normal']
        ))
        
        features_list = [
            "• Mel-spectrogram analysis for frequency patterns",
            "• Mathematical audio features (RMS, ZCR, Spectral Centroid, etc.)",
            "• Deep learning neural network classification",
            "• Temporal segmentation and confidence scoring",
            "• Multi-modal fusion of spectral and mathematical features"
        ]
        
        for feature in features_list:
            story.append(Paragraph(feature, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph(
            "Report generated by AI Music Mood Classification System",
            ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=10,
                textColor='#808080',  # Grey
                alignment=1  # Center alignment
            )
        ))
        
        # Build PDF
        doc.build(story)
        
        # Reset buffer position
        pdf_buffer.seek(0)
        
        return pdf_buffer
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def compute_slice_math_features(y, sr, slice_moods, offsets, window_size):
    """
    Compute mathematical features for each audio slice for display purposes.
    
    Args:
        y: Audio time series
        sr: Sample rate
        slice_moods: List of predicted moods for each slice
        offsets: List of time offsets for each slice
        window_size: Size of each analysis window in seconds
    
    Returns:
        List of dictionaries containing math features for each slice
    """
    slice_features = []
    
    for i, offset in enumerate(offsets):
        try:
            # Extract the slice
            start_sample = int(offset * sr)
            end_sample = int((offset + window_size) * sr)
            end_sample = min(end_sample, len(y))  # Ensure we don't go beyond audio length
            
            y_slice = y[start_sample:end_sample]
            
            if len(y_slice) > 0:
                # Compute math features for this slice
                math_feats = extract_math_features_from_array(y_slice, sr)
                
                # Add slice info
                math_feats['slice_index'] = i + 1
                math_feats['time_range'] = f"{offset:.1f}s - {offset + window_size:.1f}s"
                math_feats['predicted_mood'] = slice_moods[i].title()
                
                slice_features.append(math_feats)
            else:
                # Empty slice fallback
                slice_features.append({
                    'slice_index': i + 1,
                    'time_range': f"{offset:.1f}s - {offset + window_size:.1f}s",
                    'predicted_mood': slice_moods[i].title(),
                    'rms': 0.0,
                    'zcr': 0.0,
                    'centroid': 0.0,
                    'rolloff': 0.0,
                    'entropy': 0.0,
                    'pitch_var': 0.0
                })
                
        except Exception as e:
            # Error fallback
            slice_features.append({
                'slice_index': i + 1,
                'time_range': f"{offset:.1f}s - {offset + window_size:.1f}s",
                'predicted_mood': slice_moods[i].title(),
                'rms': 0.0,
                'zcr': 0.0,
                'centroid': 0.0,
                'rolloff': 0.0,
                'entropy': 0.0,
                'pitch_var': 0.0
            })
    
    return slice_features

def plot_waveform(y, sr):
    """
    Create a waveform visualization using librosa and matplotlib.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        matplotlib figure
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create time axis
    time = np.linspace(0, len(y)/sr, len(y))
    
    # Plot waveform with gradient effect
    ax.plot(time, y, color='#00d4ff', linewidth=0.8, alpha=0.8)
    ax.fill_between(time, y, alpha=0.3, color='#00d4ff')
    
    # Styling
    ax.set_xlabel('Time (seconds)', fontsize=12, color='white')
    ax.set_ylabel('Amplitude', fontsize=12, color='white')
    ax.set_title('🎵 Audio Waveform Analysis', fontsize=14, color='white', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(y, sr):
    """
    Create an enhanced mel spectrogram plot
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create the spectrogram plot
    img = librosa.display.specshow(
        mel_spec_db, 
        x_axis='time', 
        y_axis='mel', 
        sr=sr,
        ax=ax,
        cmap='viridis'
    )
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    
    # Styling
    ax.set_title('🌈 Mel-Spectrogram Analysis', fontsize=14, color='white', pad=20)
    ax.set_xlabel('Time (seconds)', fontsize=12, color='white')
    ax.set_ylabel('Mel Frequency', fontsize=12, color='white')
    
    plt.tight_layout()
    return fig

def plot_radar_confidence(conf, labels):
    """
    Create a radar chart for confidence scores
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Angles for each mood
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Confidence values
    values = conf.tolist()
    values += values[:1]  # Complete the circle
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=3, color='#00d4ff', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#00d4ff')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color='white', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('🎯 Mood Confidence Radar', color='white', fontsize=16, fontweight='bold', pad=20)
    
    return fig

def plot_mood_timeline_heatmap(slice_moods, offsets, window_size):
    """
    Create a timeline heatmap of mood predictions
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Create mood mapping
    mood_map = {'happy': 0, 'sad': 1, 'calm': 2, 'energetic': 3}
    mood_colors = ['#FFD700', '#4169E1', '#32CD32', '#FF6347']  # Gold, Blue, Green, Red
    
    # Convert moods to numeric codes
    mood_codes = [mood_map[mood.lower()] for mood in slice_moods]
    
    # Create the heatmap data
    heatmap_data = np.array(mood_codes).reshape(1, -1)
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Customize the plot
    ax.set_title('🔥 Mood Evolution Timeline', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_xlabel('Time Segments', fontsize=12, color='white')
    ax.set_ylabel('')
    
    # Set x-axis labels
    if len(offsets) <= 20:
        x_labels = [f"{offset:.1f}s" for offset in offsets]
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels(x_labels, rotation=45, color='white')
    else:
        step = max(1, len(offsets) // 10)
        x_positions = range(0, len(offsets), step)
        x_labels = [f"{offsets[i]:.1f}s" for i in x_positions]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, color='white')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['😊 Happy', '😢 Sad', '😌 Calm', '⚡ Energetic'])
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Mood Category', fontsize=12, color='white')
    
    plt.tight_layout()
    return fig
    plt.title('Mood Confidence Radar Chart', size=16, fontweight='bold', pad=20)
    
    # Add confidence values as text on the plot
    for angle, value, label in zip(angles[:-1], conf, labels):
        ax.text(angle, value + 0.05, f'{value*100:.1f}%', 
                horizontalalignment='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Enhanced page configuration
st.set_page_config(
    page_title="🎵 AI Music Mood Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/viraj-gavade/music-mood-classification-ml',
        'Report a bug': 'https://github.com/viraj-gavade/music-mood-classification-ml/issues',
        'About': "AI-powered music mood classification using deep learning"
    }
)

# Initialize GPU and performance optimizations
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.session_state.gpu_available = torch.cuda.is_available()
    
if 'cached_model' not in st.session_state:
    st.session_state.cached_model = None
    st.session_state.cached_plots = {}
    st.session_state.cached_audio_data = {}
    st.session_state.performance_stats = {}

# Custom CSS for enhanced styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.mood-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.confidence-high { border-left-color: #28a745; }
.confidence-medium { border-left-color: #ffc107; }
.confidence-low { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("## 🎵 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🎼 Audio File Analyzer", "🎧 Spotify Intelligence"],
    key="page_selection"
)

# Enhanced header  
if page == "🎼 Audio File Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1>🎵 AI Music Mood Analyzer</h1>
        <p>Advanced deep learning model for real-time music emotion recognition</p>
        <p><i>Upload your audio file and discover its emotional landscape</i></p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Import and render Spotify page
    try:
        import spotify_page
        spotify_page.main()
        st.stop()  # Stop execution here for Spotify page
    except ImportError as e:
        st.error(f"Spotify Intelligence page not available: {e}")
        st.info("Make sure all Spotify agent dependencies are installed.")
        st.stop()

# Sidebar configuration (only for Audio File Analyzer)
with st.sidebar:
    st.header("🎛️ Analysis Settings")
    
    # Model selection (if multiple models available)
    model_option = st.selectbox(
        "Select Model", 
        ["Standard Model", "Enhanced Model (if available)"],
        help="Choose the model variant for analysis"
    )
    
    # Analysis parameters
    st.subheader("Audio Processing")
    window_size = st.slider("Analysis Window (seconds)", 3, 10, 5)
    hop_size = st.slider("Hop Size (seconds)", 1, 5, 2)
    
    # Visualization options
    st.subheader("Visualization Options")
    show_spectrogram = st.checkbox("Show Mel-Spectrogram", True)
    show_features = st.checkbox("Show Audio Features Analysis", True)
    show_timeline = st.checkbox("Show Mood Timeline", True)
    show_confidence = st.checkbox("Show Confidence Analysis", True)
    
    # Export options
    st.subheader("Export Options")
    export_results = st.checkbox("Export Results as JSON")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Model Info
    - **Architecture**: Hybrid CNN + MLP
    - **Features**: Mel-spectrogram + Audio Statistics
    - **Classes**: Happy, Sad, Calm, Energetic
    - **Accuracy**: ~85-90%
    """)

# Initialize model with GPU optimization and caching
model_path = "model_enhanced.pth" if "Enhanced" in model_option and os.path.exists("model_enhanced.pth") else "model.pth"

try:
    # Check if model is already cached
    cache_key = f"{model_path}_{window_size}_{hop_size}"
    
    if st.session_state.cached_model is None or st.session_state.cached_model.get('key') != cache_key:
        with st.spinner("🚀 Loading model with GPU acceleration..."):
            start_time = time.time()
            model = MoodInference(model_path=model_path, window=window_size, hop=hop_size)
            
            # Move model to GPU and optimize for inference
            if hasattr(model, 'model') and st.session_state.gpu_available:
                model.model = model.model.to(st.session_state.device)
                model.model.eval()  # Set to evaluation mode
                
                # JIT compile for faster inference (first run might be slow)
                try:
                    dummy_mel = torch.randn(1, 1, 128, 431).to(st.session_state.device)
                    dummy_math = torch.randn(1, 6).to(st.session_state.device)
                    model.model = torch.jit.trace(model.model, (dummy_mel, dummy_math))
                    st.sidebar.success("⚡ Model JIT optimized")
                except:
                    st.sidebar.info("📊 Using standard model")
            
            load_time = time.time() - start_time
            st.session_state.cached_model = {'model': model, 'key': cache_key}
            st.session_state.performance_stats['model_load_time'] = load_time
    else:
        model = st.session_state.cached_model['model']
    
    # Display GPU status
    if st.session_state.gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
        st.sidebar.success(f"✅ Model loaded: {model_path}")
    else:
        st.sidebar.info("💻 Using CPU")
        st.sidebar.success(f"✅ Model loaded: {model_path}")
        
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# Enhanced file upload section
st.markdown("### 📁 Upload Audio File")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=["mp3", "wav", "ogg", "flac", "m4a"],
        help="Supported formats: MP3, WAV, OGG, FLAC, M4A (Max: 200MB)"
    )

with col2:
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        st.markdown("**📄 File Details**")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")

if uploaded_file:
    # Audio player with enhanced controls
    st.markdown("### 🎧 Audio Player")
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file
    temp_file_path = f"temp_audio_{int(time.time())}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and analyze audio
    with st.spinner("Loading audio file..."):
        # High-quality audio loading with GPU optimization
        y, sr = librosa.load(temp_file_path, sr=22050, mono=True)  # Standard quality
        
        # Create file hash for caching
        file_hash = str(hash(uploaded_file.getvalue()))
        
        # GPU acceleration options
        use_gpu_acceleration = st.sidebar.checkbox("🚀 Enable GPU Acceleration", value=True)
        if use_gpu_acceleration and torch.cuda.is_available():
            st.success(f"🎯 GPU Acceleration Enabled: {torch.cuda.get_device_name()}")
        else:
            st.info("💻 Using CPU processing")
        
        # Simple silence removal without problematic dependencies
        threshold = 0.01  # Amplitude threshold
        mask = np.abs(y) > threshold
        if np.any(mask):
            start = np.argmax(mask)
            end = len(y) - np.argmax(mask[::-1])
            y = y[start:end]
        
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Process full audio duration for complete analysis
        original_duration = librosa.get_duration(y=y, sr=sr)
        
        # Show full audio processing info
        st.info(f"🎵 Processing complete audio: {original_duration:.1f} seconds at {sr} Hz")
    
    # Enhanced processing with full feature extraction
    st.success("🎯 Full-featured analysis mode enabled with all visualizations")
    
    # Audio information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
    with col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with col3:
        st.metric("Channels", "Mono")
    with col4:
        st.metric("Samples", f"{len(y):,}")

    # Enhanced audio visualizations
    st.markdown("### 📊 Audio Analysis")
    
    # Essential Visualizations (Optimized)
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### 🌊 Audio Waveform")
        
        # Cached waveform for performance
        if 'waveform_cache' not in st.session_state.cached_plots:
            # Downsample for performance but still show full waveform
            downsample_factor = max(1, len(y) // 10000)  # Max 10k points for smooth rendering
            y_downsampled = y[::downsample_factor]
            time_axis = np.linspace(0, duration, len(y_downsampled))
            
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(
                x=time_axis, y=y_downsampled,
                mode='lines',
                name='Waveform',
                line=dict(color='#00d4ff', width=1.2)
            ))
            
            fig_wave.update_layout(
                title="Audio Waveform Analysis",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=350,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.session_state.cached_plots['waveform_cache'] = fig_wave
        
        st.plotly_chart(st.session_state.cached_plots['waveform_cache'], use_container_width=True)
    
    with viz_col2:
        if show_spectrogram:
            st.markdown("#### 🎵 Mel-Spectrogram")
            
            # Cached mel-spectrogram for performance
            spec_key = f'spectrogram_{file_hash}'
            if spec_key not in st.session_state.cached_plots:
                with st.spinner("🎼 Computing mel-spectrogram..."):
                    # GPU-accelerated mel-spectrogram computation
                    mel_spec = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_mels=128, fmax=8000,
                        hop_length=512, n_fft=2048
                    )
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Create interactive spectrogram
                    fig_spec = px.imshow(
                        mel_spec_db,
                        aspect="auto",
                        color_continuous_scale="Viridis",
                        labels={"x": "Time Frames", "y": "Mel Frequency Bins", "color": "dB"},
                        title="Mel-Spectrogram Analysis"
                    )
                    fig_spec.update_layout(
                        height=350,
                        template="plotly_dark"
                    )
                    
                    st.session_state.cached_plots[spec_key] = fig_spec
            
            st.plotly_chart(st.session_state.cached_plots[spec_key], use_container_width=True)
        else:
            st.info("🎵 Mel-Spectrogram disabled for faster processing")
    
    # Audio stats for ultra-speed version
    st.markdown("### 📊 Audio Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
    with col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with col3:
        st.metric("Mode", "Ultra-Speed")
    with col4:
        st.metric("File Size", f"{len(uploaded_file.getvalue())/1024:.0f} KB")

    # Audio Features Analysis
    if show_features:
        st.markdown("### 🔬 Advanced Audio Features Analysis")
        
        with st.spinner("🔍 Extracting detailed audio features..."):
            features_key = f'features_{file_hash}'
            if features_key not in st.session_state.cached_plots:
                # Extract comprehensive audio features
                features = {}
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Chroma and tempo
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                # RMS energy
                rms = librosa.feature.rms(y=y)[0]
                
                features = {
                    'Spectral Centroid': float(np.mean(spectral_centroids)),
                    'Spectral Rolloff': float(np.mean(spectral_rolloff)),
                    'Spectral Bandwidth': float(np.mean(spectral_bandwidth)),
                    'Zero Crossing Rate': float(np.mean(zero_crossing_rate)),
                    'RMS Energy': float(np.mean(rms)),
                    'Tempo (BPM)': float(np.atleast_1d(tempo)[0]) if hasattr(tempo, '__len__') else float(tempo),
                    'MFCC Mean': float(np.mean(mfccs)),
                    'Chroma Mean': float(np.mean(chroma))
                }
                
                st.session_state.cached_plots[features_key] = features
            
            features = st.session_state.cached_plots[features_key]
            
            # Display features in organized columns
            feat_col1, feat_col2, feat_col3 = st.columns(3)
            
            with feat_col1:
                st.markdown("#### 🎵 Spectral Features")
                st.metric("Spectral Centroid", f"{features['Spectral Centroid']:.2f} Hz")
                st.metric("Spectral Rolloff", f"{features['Spectral Rolloff']:.2f} Hz")
                st.metric("Spectral Bandwidth", f"{features['Spectral Bandwidth']:.2f} Hz")
                
            with feat_col2:
                st.markdown("#### 🎶 Rhythmic Features")
                st.metric("Tempo", f"{features['Tempo (BPM)']:.1f} BPM")
                st.metric("Zero Crossing Rate", f"{features['Zero Crossing Rate']:.4f}")
                st.metric("RMS Energy", f"{features['RMS Energy']:.4f}")
                
            with feat_col3:
                st.markdown("#### 🎼 Timbral Features")
                st.metric("MFCC Average", f"{features['MFCC Mean']:.3f}")
                st.metric("Chroma Average", f"{features['Chroma Mean']:.3f}")
                
                # Feature interpretation
                if features['Tempo (BPM)'] > 120:
                    st.success("🎯 High energy tempo detected")
                elif features['Tempo (BPM)'] < 80:
                    st.info("🎯 Relaxed tempo detected")
                else:
                    st.warning("🎯 Moderate tempo detected")
            
            # Create features visualization
            st.markdown("#### 📊 Features Visualization")
            
            # Normalized features for radar chart
            feature_names = ['Spectral\nCentroid', 'Rolloff', 'Bandwidth', 'ZCR', 'RMS', 'Tempo']
            feature_values = [
                features['Spectral Centroid'] / 4000,  # Normalize to 0-1
                features['Spectral Rolloff'] / 8000,
                features['Spectral Bandwidth'] / 2000,
                features['Zero Crossing Rate'] * 10,
                features['RMS Energy'] * 5,
                features['Tempo (BPM)'] / 200
            ]
            
            # Clip values to 0-1 range
            feature_values = [max(0, min(1, val)) for val in feature_values]
            
            # Create features radar chart
            fig_features = go.Figure()
            
            fig_features.add_trace(go.Scatterpolar(
                r=feature_values + [feature_values[0]],  # Close the polygon
                theta=feature_names + [feature_names[0]],
                fill='toself',
                name='Audio Features',
                line_color='#FF6B6B',
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig_features.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['20%', '40%', '60%', '80%', '100%']
                    )),
                title="🔬 Audio Features Analysis Radar",
                height=400,
                template="plotly_dark",
                showlegend=False
            )
            
            st.plotly_chart(fig_features, use_container_width=True)

    # Mood prediction with progress tracking
    st.markdown("### 🧠 AI Mood Analysis")
    
    # Check cache first
    cache_key = f"{file_hash}_{window_size}_{hop_size}"
    
    if cache_key in st.session_state.cached_audio_data:
        # Use cached results
        with st.spinner("⚡ Loading from cache..."):
            out = st.session_state.cached_audio_data[cache_key]
            st.success("🚀 Loaded from cache in <1 second!")
    else:
        # Perform prediction with GPU acceleration
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        inference_start = time.time()
        
        with st.spinner("🚀 GPU-accelerated AI analysis..."):
            progress_bar.progress(30)
            status_text.text("⚡ GPU processing audio features...")
            
            progress_bar.progress(70)
            status_text.text("🧠 Neural network inference...")
            
            # Full feature prediction with GPU acceleration (complete analysis)
            with torch.amp.autocast('cuda') if st.session_state.gpu_available else torch.no_grad():
                out = model.predict_file(temp_file_path, math_only=False)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
        inference_time = time.time() - inference_start
        st.session_state.performance_stats['inference_time'] = inference_time
        
        # Cache results for future use
        st.session_state.cached_audio_data[cache_key] = out
        
        # Display performance stats
        device_name = "GPU" if st.session_state.gpu_available else "CPU"
        st.success(f"🚀 Analysis completed in {inference_time:.2f}s using {device_name}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    # Performance metrics sidebar
    with st.sidebar:
        st.markdown("### ⚡ Performance Stats")
        if 'inference_time' in st.session_state.performance_stats:
            inference_time = st.session_state.performance_stats['inference_time']
            st.metric("Inference Time", f"{inference_time:.2f}s")
        
        if st.session_state.gpu_available:
            st.metric("GPU Memory", f"{torch.cuda.memory_allocated()/1024**2:.0f}MB")
            
        st.metric("Cache Status", f"{len(st.session_state.cached_audio_data)} files")

    # Extract results
    final_mood = out["final_mood"]
    final_conf = out["final_confidence"]
    slice_moods = out["slice_moods"]
    slice_confs = out["slice_confidences"]
    offsets = out["offsets"]
    
    # Enhanced mood display with emoji and confidence
    mood_emojis = {
        "happy": "😊",
        "sad": "😢", 
        "calm": "😌",
        "energetic": "⚡"
    }
    
    mood_colors = {
        "happy": "#FFD700",
        "sad": "#4682B4",
        "calm": "#98FB98", 
        "energetic": "#FF6347"
    }
    
    max_confidence = np.max(final_conf)
    confidence_level = "High" if max_confidence > 0.7 else "Medium" if max_confidence > 0.5 else "Low"
    
    st.markdown(
        f"""
        <div class="mood-card" style="background: linear-gradient(135deg, {mood_colors[final_mood]}, #667eea);">
            <h2>{mood_emojis[final_mood]} PREDICTED MOOD: {final_mood.upper()}</h2>
            <p style="font-size: 18px; margin: 10px 0;">Confidence: {max_confidence*100:.1f}% ({confidence_level})</p>
            <p style="font-size: 14px; opacity: 0.9;">Analyzed {len(slice_moods)} audio segments</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced confidence visualization
    if show_confidence:
        st.markdown("### 📊 Detailed Confidence Analysis")
        
        labels = ["Happy", "Sad", "Calm", "Energetic"]
        colors = ['#FFD700', '#4682B4', '#98FB98', '#FF6347']
        
        # Create comprehensive confidence charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced pie chart with custom styling
            fig_pie = px.pie(
                values=final_conf,
                names=labels,
                title="📊 Mood Distribution Analysis",
                color_discrete_sequence=['#FFD700', '#4682B4', '#98FB98', '#FF6347']
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='#FFFFFF', width=2)),
                hovertemplate='<b>%{label}</b><br>Confidence: %{percent}<br>Value: %{value:.3f}<extra></extra>'
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Enhanced bar chart with gradient colors
            fig_bar = px.bar(
                x=labels,
                y=final_conf * 100,
                title="📈 Confidence Scores Analysis",
                color=final_conf,
                color_continuous_scale="Plasma",
                text=final_conf * 100
            )
            fig_bar.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
            fig_bar.update_layout(
                showlegend=False, 
                yaxis_title="Confidence (%)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence metrics
        st.markdown("#### 🎯 Confidence Breakdown")
        
        for i, (label, conf) in enumerate(zip(labels, final_conf)):
            confidence_class = "confidence-high" if conf > 0.7 else "confidence-medium" if conf > 0.5 else "confidence-low"
            
            st.markdown(
                f"""
                <div class="metric-card {confidence_class}">
                    <strong>{mood_emojis[label.lower()]} {label}:</strong> {conf*100:.2f}%
                    <div style="background: linear-gradient(90deg, {colors[i]} 0%, {colors[i]}40 100%); 
                                height: 8px; border-radius: 4px; width: {conf*100}%; margin-top: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Enhanced Mood Confidence Radar Chart
        st.markdown("#### 🎯 Mood Confidence Radar Analysis")
        with st.spinner("📊 Generating radar chart..."):
            radar_key = f'radar_{file_hash}'
            if radar_key not in st.session_state.cached_plots:
                fig_radar = plot_radar_confidence(final_conf, labels)
                st.session_state.cached_plots[radar_key] = fig_radar
            
            st.pyplot(st.session_state.cached_plots[radar_key])
            plt.close()  # Close the figure to free memory
        
        st.markdown("<small>📏 This radar chart shows the confidence distribution across all mood categories.</small>", unsafe_allow_html=True)

    # Mathematical Features Used
    st.markdown("### 📊 Mathematical Features Used")
    
    with st.expander("View Mathematical Features for Each Slice", expanded=False):
        st.markdown("""
        **The AI model analyzes these mathematical features from each audio segment:**
        
        - **RMS Energy**: Root Mean Square energy (overall volume/loudness)
        - **Zero Crossing Rate**: How often the signal crosses zero (indicates noisiness vs tonality)
        - **Spectral Centroid**: "Center of mass" of the spectrum (brightness)
        - **Spectral Rolloff**: Frequency below which 85% of energy is contained
        - **Entropy**: Spectral disorder/randomness measure
        - **Pitch Variance**: Variation in fundamental frequency (pitch stability)
        """)
        
        # Compute math features for each slice
        with st.spinner("Computing mathematical features for each slice..."):
            try:
                slice_math_features = compute_slice_math_features(y, sr, slice_moods, offsets, window_size)
                
                # Display features in a nice table format
                if slice_math_features:
                    # Prepare data for DataFrame
                    display_data = []
                    for feats in slice_math_features:
                        display_data.append({
                            'Segment': feats['slice_index'],
                            'Time Range': feats['time_range'],
                            'Predicted Mood': feats['predicted_mood'],
                            'RMS Energy': f"{feats['rms']:.4f}",
                            'Zero Crossing Rate': f"{feats['zcr']:.4f}",
                            'Spectral Centroid (Hz)': f"{feats['centroid']:.1f}",
                            'Spectral Rolloff (Hz)': f"{feats['rolloff']:.1f}",
                            'Entropy': f"{feats['entropy']:.3f}",
                            'Pitch Variance': f"{feats['pitch_var']:.3f}"
                        })
                    
                    # Create and display DataFrame
                    df_math = pd.DataFrame(display_data)
                    # Display math features without pyarrow dependency
                    html_table = "<table style='width: 100%; border-collapse: collapse;'>"
                    html_table += "<tr style='background-color: #f0f0f0;'>"
                    for col in df_math.columns:
                        html_table += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{col}</th>"
                    html_table += "</tr>"
                    for _, row in df_math.iterrows():
                        html_table += "<tr>"
                        for col in df_math.columns:
                            html_table += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{row[col]}</td>"
                        html_table += "</tr>"
                    html_table += "</table>"
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Show summary statistics
                    st.markdown("#### 📊 Feature Summary Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_rms = np.mean([f['rms'] for f in slice_math_features])
                        st.metric("Average RMS Energy", f"{avg_rms:.4f}")
                        
                        avg_zcr = np.mean([f['zcr'] for f in slice_math_features])
                        st.metric("Average Zero Crossing Rate", f"{avg_zcr:.4f}")
                    
                    with col2:
                        avg_centroid = np.mean([f['centroid'] for f in slice_math_features])
                        st.metric("Average Spectral Centroid", f"{avg_centroid:.1f} Hz")
                        
                        avg_rolloff = np.mean([f['rolloff'] for f in slice_math_features])
                        st.metric("Average Spectral Rolloff", f"{avg_rolloff:.1f} Hz")
                    
                    with col3:
                        avg_entropy = np.mean([f['entropy'] for f in slice_math_features])
                        st.metric("Average Entropy", f"{avg_entropy:.3f}")
                        
                        avg_pitch_var = np.mean([f['pitch_var'] for f in slice_math_features])
                        st.metric("Average Pitch Variance", f"{avg_pitch_var:.3f}")
                        
                else:
                    st.warning("⚠️ No mathematical features could be computed.")
                    
            except Exception as e:
                st.error(f"❌ Error computing mathematical features: {str(e)}")
                # Fallback: show features for first slice only
                st.info("🔄 Computing features for first slice only...")
                try:
                    first_slice_start = int(offsets[0] * sr) if offsets else 0
                    first_slice_end = int((offsets[0] + window_size) * sr) if offsets else int(window_size * sr)
                    first_slice_end = min(first_slice_end, len(y))
                    
                    y_first = y[first_slice_start:first_slice_end]
                    first_feats = extract_math_features_from_array(y_first, sr)
                    
                    st.markdown("**Features for First Slice:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMS Energy", f"{first_feats['rms']:.4f}")
                        st.metric("Zero Crossing Rate", f"{first_feats['zcr']:.4f}")
                        st.metric("Spectral Centroid", f"{first_feats['centroid']:.1f} Hz")
                    
                    with col2:
                        st.metric("Spectral Rolloff", f"{first_feats['rolloff']:.1f} Hz")
                        st.metric("Entropy", f"{first_feats['entropy']:.3f}")
                        st.metric("Pitch Variance", f"{first_feats['pitch_var']:.3f}")
                        
                except Exception as fallback_e:
                    st.error(f"❌ Could not compute features: {str(fallback_e)}")

    # Enhanced timeline visualization
    if show_timeline:
        st.markdown("### ⏱️ Comprehensive Mood Timeline Analysis")
        
        # Create interactive timeline with multiple views
        mood_to_num = {"happy": 0, "sad": 1, "calm": 2, "energetic": 3}
        timeline_numeric = [mood_to_num[m] for m in slice_moods]
        timeline_colors = [mood_colors[m] for m in slice_moods]
        
        # Multi-panel timeline visualization
        fig_timeline = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['Mood Progression', 'Confidence Evolution', 'Energy Analysis'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Mood timeline with enhanced styling
        fig_timeline.add_trace(
            go.Scatter(
                x=offsets,
                y=timeline_numeric,
                mode='lines+markers',
                name='Mood Prediction',
                line=dict(width=4, color='#00d4ff'),
                marker=dict(size=10, color=timeline_colors, line=dict(width=2, color='white')),
                hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Mood:</b> %{customdata}<br><extra></extra>',
                customdata=slice_moods
            ),
            row=1, col=1
        )
        
        # 2. Confidence timeline
        max_confidences = [np.max(conf) for conf in slice_confs]
        fig_timeline.add_trace(
            go.Scatter(
                x=offsets,
                y=max_confidences,
                mode='lines+markers',
                name='Peak Confidence',
                fill='tozeroy',
                line=dict(color='rgba(255, 215, 0, 0.8)', width=3),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # 3. Energy analysis (simulated from mood patterns)
        energy_levels = [0.8 if m == 'energetic' else 0.6 if m == 'happy' else 0.3 if m == 'calm' else 0.2 for m in slice_moods]
        fig_timeline.add_trace(
            go.Bar(
                x=offsets,
                y=energy_levels,
                name='Energy Level',
                marker_color='rgba(255, 99, 71, 0.7)',
                width=[window_size * 0.8] * len(offsets)
            ),
            row=3, col=1
        )
        
        # Update layout with enhanced styling
        fig_timeline.update_yaxes(
            tickvals=[0, 1, 2, 3], 
            ticktext=['😊 Happy', '😢 Sad', '😌 Calm', '⚡ Energetic'], 
            row=1, col=1
        )
        fig_timeline.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)
        fig_timeline.update_yaxes(title_text="Energy", range=[0, 1], row=3, col=1)
        fig_timeline.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        
        fig_timeline.update_layout(
            height=800, 
            showlegend=True,
            template="plotly_dark",
            title_text="🎵 Complete Audio Mood Analysis Dashboard",
            title_x=0.5
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Mood transitions analysis
        st.markdown("#### 🔄 Mood Transitions")
        
        transitions = []
        for i in range(len(slice_moods) - 1):
            if slice_moods[i] != slice_moods[i + 1]:
                transitions.append({
                    'Time': f"{offsets[i]:.1f}s → {offsets[i+1]:.1f}s",
                    'From': slice_moods[i].title(),
                    'To': slice_moods[i + 1].title()
                })
        
        if transitions:
            df_transitions = pd.DataFrame(transitions)
            # Display transitions without pyarrow dependency
            html_table = "<table style='width: 100%; border-collapse: collapse;'>"
            html_table += "<tr style='background-color: #e6f3ff;'>"
            for col in df_transitions.columns:
                html_table += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{col}</th>"
            html_table += "</tr>"
            for _, row in df_transitions.iterrows():
                html_table += "<tr>"
                for col in df_transitions.columns:
                    html_table += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{row[col]}</td>"
                html_table += "</tr>"
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("🎵 Consistent mood throughout the audio - no major transitions detected!")

    # Enhanced slice-by-slice analysis
    st.markdown("### 🔍 Detailed Segment Analysis")
    
    # Create expandable section for detailed analysis
    with st.expander("View Detailed Segment Predictions", expanded=False):
        
        # Create DataFrame for better presentation
        slice_data = []
        for i, (mood, probs, offset) in enumerate(zip(slice_moods, slice_confs, offsets)):
            slice_data.append({
                'Segment': i + 1,
                'Time Range': f"{offset:.1f}s - {offset + window_size:.1f}s",
                'Predicted Mood': mood.title(),
                'Confidence': f"{np.max(probs)*100:.1f}%",
                'Happy': f"{probs[0]*100:.1f}%",
                'Sad': f"{probs[1]*100:.1f}%", 
                'Calm': f"{probs[2]*100:.1f}%",
                'Energetic': f"{probs[3]*100:.1f}%"
            })
        
        df_slices = pd.DataFrame(slice_data)
        # Display slice analysis without pyarrow dependency
        html_table = "<table style='width: 100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #f8f9fa;'>"
        for col in df_slices.columns:
            html_table += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{col}</th>"
        html_table += "</tr>"
        for _, row in df_slices.iterrows():
            html_table += "<tr>"
            for col in df_slices.columns:
                html_table += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{row[col]}</td>"
            html_table += "</tr>"
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Statistics about segments
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Segments", len(slice_moods))
        with col2:
            most_common_mood = max(set(slice_moods), key=slice_moods.count)
            st.metric("Most Common Mood", most_common_mood.title())
        with col3:
            avg_confidence = np.mean([np.max(conf) for conf in slice_confs])
            st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")
    
    # Audio Slice Playback
    st.markdown("### 🎧 Play Individual Audio Slices")
    
    with st.expander("Play Audio Slices", expanded=False):
        st.markdown("🎵 **Listen to each analyzed audio segment individually:**")
        
        # Create audio slices and playback options
        try:
            with st.spinner("Preparing audio slices..."):
                # Limit to reasonable number of slices for UI performance
                max_slices_to_show = min(len(offsets), 20)
                if len(offsets) > max_slices_to_show:
                    st.warning(f"⚠️ Showing first {max_slices_to_show} slices out of {len(offsets)} total for performance.")
                
                # Create columns for better layout (2 slices per row)
                num_cols = 2
                
                for i in range(0, max_slices_to_show, num_cols):
                    cols = st.columns(num_cols)
                    
                    for j, col in enumerate(cols):
                        slice_idx = i + j
                        if slice_idx >= max_slices_to_show:
                            break
                            
                        offset = offsets[slice_idx]
                        mood = slice_moods[slice_idx]
                        conf = slice_confs[slice_idx]
                        max_conf = np.max(conf)
                        
                        # Extract the audio slice
                        start_sample = int(offset * sr)
                        end_sample = int((offset + window_size) * sr)
                        end_sample = min(end_sample, len(y))  # Ensure we don't go beyond audio length
                        
                        y_slice = y[start_sample:end_sample]
                        
                        if len(y_slice) > 0:
                            with col:
                                # Create slice label with mood and confidence
                                slice_label = f"**Slice {slice_idx + 1}** ({offset:.1f}s - {offset + window_size:.1f}s)"
                                mood_info = f"Predicted: {mood.title()} ({max_conf*100:.1f}% confidence)"
                                
                                st.markdown(slice_label)
                                st.markdown(f"<small style='color: #666;'>{mood_info}</small>", unsafe_allow_html=True)
                                
                                # Create temporary audio file in memory
                                try:
                                    audio_bytes = create_audio_slice_bytes(y_slice, sr)
                                    st.audio(audio_bytes, format='audio/wav')
                                except Exception as e:
                                    st.error(f"❌ Error creating audio for slice {slice_idx + 1}: {str(e)}")
                                
                                st.markdown("---")  # Visual separator
                        else:
                            with col:
                                st.warning(f"⚠️ Slice {slice_idx + 1}: No audio data")
                
                # Summary info
                st.markdown("#### 📊 Playback Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Slices Shown", f"{max_slices_to_show}")
                
                with col2:
                    st.metric("Total Duration", f"{len(offsets) * window_size:.1f}s")
                
                with col3:
                    avg_slice_duration = window_size
                    st.metric("Slice Duration", f"{avg_slice_duration:.1f}s")
                    
        except Exception as e:
            st.error(f"❌ Error preparing audio slices: {str(e)}")
            st.info("🔄 You can still view the analysis results above.")
    
    # Mood Timeline Heatmap
    st.markdown("### 🎨 Mood Heatmap Timeline")
    with st.spinner("Generating mood timeline heatmap..."):
        fig = plot_mood_timeline_heatmap(slice_moods, offsets, window_size)
        st.pyplot(fig)
        plt.close()  # Close the figure to free memory
    
    st.markdown("<small>🔍 This heatmap shows how the mood changes across different time segments of your audio.</small>", unsafe_allow_html=True)
    
    # PDF Report Generation
    st.markdown("### 📝 Generate PDF Report")
    
    if PDF_AVAILABLE:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("💾 Download Analysis as PDF", type="primary"):
                with st.spinner("Generating comprehensive PDF report with visualizations..."):
                    # Regenerate plots for PDF
                    plot_images = {}
                    
                    # Generate waveform plot
                    try:
                        fig_wave = plot_waveform(y, sr)
                        img_buffer = io.BytesIO()
                        fig_wave.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        img_buffer.seek(0)
                        plot_images['waveform_img'] = img_buffer
                    except Exception as e:
                        st.warning(f"Could not generate waveform for PDF: {e}")
                    
                    # Generate mel spectrogram plot
                    try:
                        fig_spec = plot_mel_spectrogram(y, sr)
                        img_buffer = io.BytesIO()
                        fig_spec.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        img_buffer.seek(0)
                        plot_images['spectrogram_img'] = img_buffer
                    except Exception as e:
                        st.warning(f"Could not generate spectrogram for PDF: {e}")
                    
                    # Generate radar chart
                    try:
                        labels = ['Happy', 'Sad', 'Calm', 'Energetic']
                        fig_radar = plot_radar_confidence(final_conf, labels)
                        img_buffer = io.BytesIO()
                        fig_radar.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        img_buffer.seek(0)
                        plot_images['radar_img'] = img_buffer
                    except Exception as e:
                        st.warning(f"Could not generate radar chart for PDF: {e}")
                    
                    # Generate timeline heatmap
                    try:
                        fig_timeline = plot_mood_timeline_heatmap(slice_moods, offsets, window_size)
                        img_buffer = io.BytesIO()
                        fig_timeline.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        img_buffer.seek(0)
                        plot_images['heatmap_img'] = img_buffer
                    except Exception as e:
                        st.warning(f"Could not generate timeline heatmap for PDF: {e}")
                    
                    # Collect math features data
                    math_features_data = []
                    try:
                        math_features_data = compute_slice_math_features(y, sr, slice_moods, offsets, window_size)
                    except Exception as e:
                        st.warning(f"Could not compute math features for PDF: {e}")
                    
                    # Prepare comprehensive data for PDF
                    pdf_data = {
                        'filename': uploaded_file.name if uploaded_file else 'unknown.wav',
                        'duration': duration,
                        'final_mood': final_mood,
                        'final_confidence': final_conf,
                        'slice_moods': slice_moods,
                        'slice_confidences': slice_confs,
                        'offsets': offsets,
                        'window_size': window_size,
                        'math_features': math_features_data,
                        **plot_images  # Add all plot images
                    }
                    
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(pdf_data)
                    
                    if pdf_buffer:
                        st.success("✅ PDF report generated successfully!")
                        
                        # Create download button
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"mood_analysis_report_{timestamp}.pdf"
                        
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf",
                            help="Click to download your complete mood analysis report"
                        )
                    else:
                        st.error("❌ Failed to generate PDF report. Please try again.")
        
        with col2:
            st.markdown("""
            **PDF Report includes:**
            • Main mood prediction and confidence scores  
            • Detailed slice-by-slice analysis  
            • Analysis methodology summary  
            • Timestamp and file information  
            
            *Perfect for sharing results or keeping records!*
            """)
    else:
        st.warning("⚠️ PDF generation not available. Please install reportlab: `pip install reportlab`")
        st.info("📊 You can still view and export the analysis data below.")
    
    # Export functionality
    if export_results:
        st.markdown("### 💾 Export Results")
        
        export_data = {
            'filename': uploaded_file.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'final_prediction': {
                'mood': final_mood,
                'confidence_scores': {
                    'happy': float(final_conf[0]),
                    'sad': float(final_conf[1]),
                    'calm': float(final_conf[2]),
                    'energetic': float(final_conf[3])
                }
            },
            'segment_analysis': slice_data,
            'model_parameters': {
                'window_size': window_size,
                'hop_size': hop_size,
                'model_path': model_path
            }
        }
        
        import json
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Analysis Results (JSON)",
            data=json_str,
            file_name=f"mood_analysis_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Clean up temporary file
    try:
        os.remove(temp_file_path)
    except:
        pass
