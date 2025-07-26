import fitz  # PyMuPDF


def add_word_style_borders(input_path, output_path, border_width=1):
    doc = fitz.open(input_path)

    for page in doc:
        rect = page.rect  # Get page size
        width, height = rect.width, rect.height

        # Detect orientation
        portrait = height >= width

        # Set margins based on orientation (Word-style)
        if portrait:
            left, right = 36, 36  # 0.5 inch
            top, bottom = 30, 30  # 0.5 inch
        else:
            left, right = 22, 22  # Wider on landscape
            top, bottom = 30, 30

        # Define the border rectangle
        border_rect = fitz.Rect(left, top, width - right, height - bottom)

        # Draw border
        shape = page.new_shape()
        shape.draw_rect(border_rect)
        shape.finish(width=border_width, color=(0, 0, 0))  # Black border
        shape.commit()

    doc.save(output_path)


# Example usage
add_word_style_borders(
    "AHS 242101011 - Anu Priya - Thesis (2).pdf", "output_with_word_style_borders.pdf"
)
