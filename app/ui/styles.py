STYLE_SHEET = """
QWidget {
    font-family: "Segoe UI";
    font-size: 11pt;
    color: #e5e7eb;
    background-color: #1f1f1f;
}
#title {
    font-size: 18pt;
    font-weight: 700;
    color: #d9e2ef;
}
QLineEdit, QComboBox {
    background: #1f1f1f;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e5e7eb;
}
QComboBox::drop-down {
    subcontrol-position: center right;
    width: 20px;
    border: 1px solid #CCCCCC;
    border-radius: 2px;
    background: #f0f0f0;
    margin: 1px;
}

QComboBox::down-arrow {
    width: 0px;
    height: 0px;
    border-left: 6px solid #f0f0f0;
    border-right: 6px solid #f0f0f0;
    border-top: 8px solid #333333;
    margin-top: 2px;
}

QComboBox::down-arrow:hover {
    border-top-color: #2196F3;
}

QGroupBox {
    border: 1px solid #2f2f2f;
    border-radius: 12px;
    background: #232323;
    padding: 12px;
}
QGroupBox:title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #cbd5f5;
}
#videoFrame {
    border-radius: 10px;
    background: #151515;
    min-height: 420px;
}
#statusBox {
    background: #1c3a2d;
    border: 1px solid #256c4d;
    border-radius: 6px;
    color: #d1fae5;
}
#successTitle {
    font-size: 12pt;
    font-weight: 700;
    color: #34d399;
}
#historyTitle {
    font-size: 11pt;
    font-weight: 700;
    color: #93c5fd;
}
#hLine {
    background: #2f2f2f;
    height: 1px;
}
#avatar {
    border: 1px dashed #3a3a3a;
    border-radius: 6px;
    background: #1a1a1a;
}
#primaryButton {
    background: #2563eb;
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
}
#primaryButton:hover { background: #1d4ed8; }

#secondaryButton{
    background: #6b7280;
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
}
#secondaryButton:hover { background: #4b5563; }
#ghostButton {
    background: #2f2f2f;
    color: #e5e7eb;
    padding: 8px 16px;
    border-radius: 8px;
}
QLabel {
    background: transparent;
}
"""
