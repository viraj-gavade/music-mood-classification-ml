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
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Audio Waveform', fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_mel_spectrogram(y, sr):
    """
    Create a Mel Spectrogram visualization using librosa and matplotlib.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Display the spectrogram
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
    
    plt.title('Mel Spectrogram', fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mel Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt

def plot_mood_timeline_heatmap(slice_moods, offsets, window_size):
    """
    Create a Mood Timeline Heatmap visualization.
    
    Args:
        slice_moods: List of predicted moods for each slice
        offsets: List of time offsets for each slice
        window_size: Size of each analysis window in seconds
    
    Returns:
        matplotlib figure
    """
    # Convert moods to numeric codes
    mood_to_code = {'happy': 0, 'sad': 1, 'calm': 2, 'energetic': 3}
    mood_codes = [mood_to_code[mood.lower()] for mood in slice_moods]
    
    # Create the heatmap data (1 row, N columns)
    heatmap_data = np.array(mood_codes).reshape(1, -1)
    
    # Create figure
    plt.figure(figsize=(14, 2))
    
    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Customize the plot
    plt.title('Mood Heatmap Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Time Segments')
    plt.ylabel('')
    
    # Set x-axis labels to show time ranges
    if len(offsets) <= 20:  # Show all labels for shorter sequences
        x_labels = [f"{offset:.1f}s" for offset in offsets]
        plt.xticks(range(len(offsets)), x_labels, rotation=45)
    else:  # Show fewer labels for longer sequences
        step = len(offsets) // 10
        x_positions = range(0, len(offsets), step)
        x_labels = [f"{offsets[i]:.1f}s" for i in x_positions]
        plt.xticks(x_positions, x_labels, rotation=45)
    
    # Remove y-axis ticks
    plt.yticks([])
    
    # Add colorbar with mood labels
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.2, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Happy', 'Sad', 'Calm', 'Energetic'])
    cbar.set_label('Mood Category', fontsize=12)
    
    plt.tight_layout()
    return plt

def plot_radar_confidence(conf, labels):
    """
    Create a radar/spider chart for mood confidence scores.
    
    Args:
        conf: Array of confidence scores (4 values)
        labels: List of mood labels
    
    Returns:
        matplotlib figure
    """
    # Number of variables
    N = len(labels)
    
    # Compute angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the plot
    
    # Close the confidence values
    conf_closed = list(conf) + [conf[0]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot the confidence scores
    ax.plot(angles, conf_closed, 'o-', linewidth=2, label='Confidence', color='#667eea')
    ax.fill(angles, conf_closed, alpha=0.25, color='#667eea')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True)
    
    # Add title
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
    show_spectrogram = st.checkbox("Show Spectrogram", True)
    show_features = st.checkbox("Show Audio Features", True)
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

# Initialize model
try:
    model_path = "model_enhanced.pth" if "Enhanced" in model_option and os.path.exists("model_enhanced.pth") else "model.pth"
    model = MoodInference(model_path=model_path, window=window_size, hop=hop_size)
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
        y, sr = librosa.load(temp_file_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
    
    # Waveform Visualization
    st.markdown("### 📊 Waveform Visualization")
    with st.spinner("Generating waveform..."):
        fig = plot_waveform(y, sr)
        st.pyplot(fig)
        plt.close()  # Close the figure to free memory
    
    # Mel Spectrogram
    st.markdown("### 🎛️ Mel Spectrogram")
    with st.spinner("Generating mel spectrogram..."):
        fig = plot_mel_spectrogram(y, sr)
        st.pyplot(fig)
        plt.close()  # Close the figure to free memory
    
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
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waveform", "🎼 Spectrogram", "📈 Features", "🔊 Audio Stats"])
    
    with tab1:
        st.markdown("#### Waveform Analysis")
        
        # Interactive waveform with Plotly
        time_axis = np.linspace(0, duration, len(y))
        
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis, y=y,
            mode='lines',
            name='Waveform',
            line=dict(color='#667eea', width=1)
        ))
        
        fig_wave.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_wave, width='stretch')
    
    with tab2:
        if show_spectrogram:
            st.markdown("#### Mel-Spectrogram")
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Create interactive spectrogram
            fig_spec = px.imshow(
                mel_spec_db,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={"x": "Time Frames", "y": "Mel Frequency Bins", "color": "dB"}
            )
            fig_spec.update_layout(
                title="Mel-Spectrogram",
                height=400
            )
            
            st.plotly_chart(fig_spec, width='stretch')
    
    with tab3:
        if show_features:
            st.markdown("#### Audio Features")
            
            # Extract various audio features
            features = {}
            
            # Spectral features
            features['Spectral Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['Spectral Rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['Zero Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['RMS Energy'] = np.mean(librosa.feature.rms(y=y))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(5):  # Show first 5 MFCCs
                features[f'MFCC {i+1}'] = np.mean(mfccs[i])
            
            # Display features as metrics
            cols = st.columns(3)
            for i, (feature, value) in enumerate(features.items()):
                with cols[i % 3]:
                    st.metric(feature, f"{value:.4f}")
    
    with tab4:
        st.markdown("#### Statistical Analysis")
        
        # Audio statistics
        stats = {
            'Mean Amplitude': np.mean(np.abs(y)),
            'Max Amplitude': np.max(np.abs(y)),
            'Standard Deviation': np.std(y),
            'Dynamic Range': np.max(y) - np.min(y),
            'RMS': np.sqrt(np.mean(y**2))
        }
        
        # Create bar chart for statistics
        fig_stats = px.bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            title="Audio Statistics",
            color=list(stats.values()),
            color_continuous_scale="Blues"
        )
        fig_stats.update_layout(height=400)
        st.plotly_chart(fig_stats, width='stretch')

    # Mood prediction with progress tracking
    st.markdown("### 🧠 AI Mood Analysis")
    
    # Progress bar for prediction
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🤖 AI is analyzing the emotional content..."):
        progress_bar.progress(20)
        status_text.text("Extracting audio features...")
        time.sleep(0.5)
        
        progress_bar.progress(50)
        status_text.text("Processing through neural network...")
        
        out = model.predict_file(temp_file_path)
        
        progress_bar.progress(80)
        status_text.text("Generating confidence scores...")
        time.sleep(0.3)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete! ✨")
        time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

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
        
        # Create confidence charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=final_conf,
                names=labels,
                title="Mood Distribution",
                color_discrete_sequence=colors
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=labels,
                y=final_conf * 100,
                title="Confidence Scores (%)",
                color=final_conf,
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(showlegend=False, yaxis_title="Confidence (%)")
            st.plotly_chart(fig_bar, width='stretch')
        
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
        
        # Mood Confidence Radar Chart
        st.markdown("#### 🎯 Mood Confidence Radar Chart")
        with st.spinner("Generating radar chart..."):
            fig = plot_radar_confidence(final_conf, labels)
            st.pyplot(fig)
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
        st.markdown("### ⏱️ Mood Timeline Analysis")
        
        # Create interactive timeline
        mood_to_num = {"happy": 0, "sad": 1, "calm": 2, "energetic": 3}
        timeline_numeric = [mood_to_num[m] for m in slice_moods]
        timeline_colors = [mood_colors[m] for m in slice_moods]
        
        # Timeline with confidence overlay
        fig_timeline = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Mood Timeline', 'Confidence Over Time'],
            vertical_spacing=0.1
        )
        
        # Mood timeline
        fig_timeline.add_trace(
            go.Scatter(
                x=offsets,
                y=timeline_numeric,
                mode='lines+markers',
                name='Mood',
                line=dict(width=3),
                marker=dict(size=8, color=timeline_colors)
            ),
            row=1, col=1
        )
        
        # Confidence timeline
        max_confidences = [np.max(conf) for conf in slice_confs]
        fig_timeline.add_trace(
            go.Scatter(
                x=offsets,
                y=max_confidences,
                mode='lines+markers',
                name='Confidence',
                fill='tonexty',
                line=dict(color='rgba(102, 126, 234, 0.8)', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig_timeline.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=['Happy', 'Sad', 'Calm', 'Energetic'], row=1, col=1)
        fig_timeline.update_yaxes(title_text="Confidence", row=2, col=1)
        fig_timeline.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig_timeline.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig_timeline, width='stretch')
        
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
