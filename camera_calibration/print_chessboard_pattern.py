import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# Configuration
inner_corners_x = 9
inner_corners_y = 6
squares_x = inner_corners_x + 1  # 10
squares_y = inner_corners_y + 1  # 7
square_size_mm = 29.7  # Slightly reduced to fit A4 landscape

# Total board size in mm
board_width_mm = squares_x * square_size_mm
board_height_mm = squares_y * square_size_mm

# Convert mm to inches for matplotlib (1 inch = 25.4 mm)
board_width_in = board_width_mm / 25.4
board_height_in = board_height_mm / 25.4

# Create the figure
fig, ax = plt.subplots(figsize=(board_width_in, board_height_in))
ax.set_xlim(0, board_width_mm)
ax.set_ylim(0, board_height_mm)
ax.set_aspect('equal')
ax.axis('off')

# Draw the chessboard pattern
for i in range(squares_x):
    for j in range(squares_y):
        if (i + j) % 2 == 0:
            color = 'black'
        else:
            color = 'white'
        rect = Rectangle((i * square_size_mm, j * square_size_mm),
                         square_size_mm, square_size_mm,
                         facecolor=color, edgecolor='black', linewidth=0.1)
        ax.add_patch(rect)

# Save to PDF
pdf_filename = "calibration_chessboard_9x6_A4_landscape.pdf"
with PdfPages(pdf_filename) as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.close(fig)
