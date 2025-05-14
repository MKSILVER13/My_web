import networkx as nx
from pyvis.network import Network
import fitz  # PyMuPDF
import os
import re
import sys
import json  # new import for JSON output
import pytesseract  # new import for OCR
from PIL import Image  # new import for OCR image processing

class PDFKnowledgeGraphV2:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def ocr_force(self, doc):
        """
        Extract text from PDF with formatting information using PyMuPDF.
        Uses geometric analysis to detect underlined text.
        Returns a list of dictionaries with text and its formatting properties.
        """
        try:
            # print(f"Opening PDF file: {pdf_path}")
            doc = fitz.open(pdf_path)
            # print(f"PDF has {len(doc)} pages")
            
            # Process document with OCR for pages that need it
            processed_doc = self.ocr_force(doc)
            
            structured_text = []
            
            # First pass: determine maximum font size for reference
            max_font_size = 0
            # Iterate through the original document for font size calculation
            for page_idx in range(len(doc)):
                page_content = doc.load_page(page_idx)
                blocks = page_content.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                max_font_size = max(max_font_size, span["size"])
            # print(f"Maximum font size detected: {max_font_size}")
              # Process each page: extract text spans and detect underlines via drawing objects.
            for page_num_idx, item in enumerate(processed_doc):
                # Check if this is an OCR result (dictionary) or a regular page
                if isinstance(item, dict) and "source" in item and item["source"] in ["ocr", "ocr-error"]:
                    # This is an OCR result, add it directly to structured text
                    structured_text.append(item)
                    continue
                
                # This is a regular page, process it normally
                page_content_to_process = item                # Get drawing objects and filter for those that appear to be underlines.
                drawings = page_content_to_process.get_drawings()
                underline_rects = []
                for d in drawings:
                    # Look for filled rectangles (type 'f') which might be drawn as underlines.
                    if d.get("type") == "f":
                        for drawing_item in d.get("items", []):
                            if drawing_item[0] == "re":
                                rect = drawing_item[1]
                                # Heuristic: if the rectangle is very short in height, consider it an underline.
                                if rect.height < 5:
                                    underline_rects.append(rect)
                
                blocks = page_content_to_process.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            is_bold = False
                            is_underlined = False
                            font_size = 0
                            font_used = ""
                            span_bboxes = []
                            
                            # Process each span within the line.
                            for span in line["spans"]:
                                if "bold" in span["font"].lower():
                                    is_bold = True
                                line_text += span["text"] + " "
                                font_size = max(font_size, span["size"])
                                font_used = span["font"]
                                span_bboxes.append(span["bbox"])
                            
                            # Compute the union of the bounding boxes for the whole line.
                            if span_bboxes:
                                x0 = min(b[0] for b in span_bboxes)
                                y0 = min(b[1] for b in span_bboxes)
                                x1 = max(b[2] for b in span_bboxes)
                                y1 = max(b[3] for b in span_bboxes)
                                line_bbox = (x0, y0, x1, y1)
                            else:
                                line_bbox = None
                            
                            # Heuristic: if any underline rectangle overlaps horizontally
                            # and its top is within 5 units of the text bbox bottom, mark as underlined.
                            if line_bbox:
                                for rect in underline_rects:
                                    overlap = min(line_bbox[2], rect.x1) - max(line_bbox[0], rect.x0)
                                    if overlap > 0 and abs(rect.y0 - line_bbox[3]) < 5:
                                        is_underlined = True
                                        break
                            
                            line_text = line_text.strip()
                            if line_text:                                structured_text.append({
                                    "text": line_text,
                                    "font": font_used,
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                    "is_underlined": is_underlined,
                                    "page": page_content_to_process.number + 1,
                                    "source": "text"  # Mark as normal text extraction
                                })
            
            # print(f"Extracted {len(structured_text)} text elements")
            return structured_text
        except Exception as e:
            # print(f"Error extracting PDF text: {e}")
            raise e

    def build_knowledge_graph(self, elements, pdf_name):
        """
        Build a knowledge graph from the structured text elements.
        Ranking for bold nodes (headings) is determined as follows:
          - Larger font size → Higher priority (closer to the root).
          - Underlined bold text > Non-underlined bold text (if same font size).
          - For equal font sizes and same underlining, nodes become siblings (new node’s parent is determined from one level up).
          - If new bold node has a font size greater or equal to the candidate parent, traverse up until finding a node with a strictly greater font size.
        """
        try:
            # Create root node based on the PDF filename.
            root = os.path.splitext(os.path.basename(pdf_name))[0]
            # print(f"Root node: {root}")
            self.graph.add_node(root, level=0, content="", font_size=100, is_underlined=False)
            
            current_node = root
            node_stack = [root]
            
            # Skip the first bold text (typically the title).
            first_bold_skipped = False
            
            # Determine a reference regular (non-bold) text font size.
            regular_font_size = None
            for elem in elements:
                if not elem["is_bold"]:
                    regular_font_size = elem["font_size"]
                    # print(f"Regular text font size detected: {regular_font_size}")
                    break
            if regular_font_size is None:
                # print("Warning: No regular text font size found, using 8.0 as default")
                regular_font_size = 8.0
            
            # print(f"Starting to process {len(elements)} text elements")
            nodes_added = 0
            
            # Track if the next text starts with a colon.
            next_starts_with_colon = False
            
            for i, elem in enumerate(elements):
                text = elem["text"]
                is_bold = elem["is_bold"]
                is_underlined = elem["is_underlined"]
                font_size = elem["font_size"]
                
                # Check if the next element (if exists) starts with a colon.
                if i < len(elements) - 1 and not elements[i+1]["is_bold"]:
                    next_text = elements[i+1]["text"]
                    next_starts_with_colon = next_text.startswith(":")
                else:
                    next_starts_with_colon = False
                
                # print(f"Processing: '{text[:30]}...' - Bold: {is_bold}, Underlined: {is_underlined}, Font: {font_size}, Regular Font: {regular_font_size}")
                
                # Skip the very first bold text (title).
                if is_bold and not first_bold_skipped:
                    first_bold_skipped = True
                    # print(f"Skipping first bold text: {text}")
                    continue
                
                # Skip standalone numbered items.
                if re.match(r'^\d+\.\s*$', text):
                    # print(f"Skipping numbered item: {text}")
                    continue
                
                if is_bold:
                    # Determine if the bold text qualifies as a heading.
                    is_heading = (font_size > regular_font_size) or text.endswith(':') or next_starts_with_colon
                    if is_heading:
                        # --- Parent Selection Logic ---
                        while len(node_stack) > 0:
                            candidate = node_stack[-1]
                            candidate_attrs = self.graph.nodes[candidate]
                            candidate_fs = candidate_attrs["font_size"]
                            candidate_underlined = candidate_attrs.get("is_underlined", False)
                            
                            # Accept candidate if it is the root or has a strictly greater font size.
                            if candidate == root or candidate_fs > font_size:
                                break
                            # For equal font sizes:
                            if candidate_fs == font_size:
                                # If both have identical underlining, pop candidate so new node becomes sibling.
                                if candidate_underlined == is_underlined:
                                    node_stack.pop()
                                # If candidate is underlined and new node is not, candidate remains.
                                elif candidate_underlined and not is_underlined:
                                    break
                                # If candidate is non-underlined but new node is underlined, pop candidate.
                                elif not candidate_underlined and is_underlined:
                                    node_stack.pop()
                                else:
                                    break
                            # If candidate's font size is smaller, pop it.
                            if candidate_fs < font_size:
                                node_stack.pop()
                            if not node_stack:
                                break
                        if not node_stack:
                            parent_node = root
                            node_stack.append(root)
                        else:
                            parent_node = node_stack[-1]
                        # --- End Parent Selection Logic ---
                        
                        # print(f"Parent node determined: {parent_node}")
                        
                        new_node_id = text  # You can append page info if desired.
                        self.graph.add_node(new_node_id, level=len(node_stack), content="", font_size=font_size, is_underlined=is_underlined)
                        self.graph.add_edge(parent_node, new_node_id)
                        
                        current_node = new_node_id
                        node_stack.append(new_node_id)
                        nodes_added += 1
                        # print(f"Added node: {new_node_id} under parent: {parent_node}")
                    else:
                        # Bold text that doesn't qualify as a heading is added as content.
                        # print(f"Adding bold text as content to {current_node}: {text}")
                        if current_node in self.graph.nodes:
                            self.graph.nodes[current_node]["content"] += " " + text
                else:
                    # Non-bold text is added as content to the current node.
                    # print(f"Adding regular text as content to {current_node}: {text}")
                    if current_node in self.graph.nodes:
                        self.graph.nodes[current_node]["content"] += " " + text
            
            # Clean up extra whitespace from node content.
            for node in self.graph.nodes():
                if "content" in self.graph.nodes[node]:
                    content = self.graph.nodes[node]["content"].strip()
                    self.graph.nodes[node]["content"] = content if content else "(No description)"
            
            # print(f"Created graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            # print(f"Added {nodes_added} heading nodes from text")
        except Exception as e:
            print(f"Error building knowledge graph: {e}")
            raise e

    def visualize(self, output_file):
        """
        Visualize the knowledge graph using PyVis with color coding:
          - Different colors for different levels.
          - Underlined nodes get a special gold color.
        """
        try:
            net = Network(height="750px", width="100%", directed=True)
            colors = {
                0: '#ff0000',  # Root - Red
                1: '#00cc00',  # Level 1 - Green
                2: '#0099ff',  # Level 2 - Blue
                3: '#9933ff',  # Level 3 - Purple
                4: '#ff9900',  # Level 4 - Orange
            }
            for node in self.graph.nodes():
                attrs = self.graph.nodes[node]
                level = attrs.get("level", 0)
                color = colors.get(level, "#cccccc")
                if attrs.get("is_underlined", False):
                    color = "#FFD700"  # Gold for underlined nodes
                label = node if len(node) <= 30 else node[:30] + "..."
                content = attrs.get("content", "")
                title = f"{node}\n\n{content}" if content else node
                net.add_node(node, label=label, title=title, color=color)
            for edge in self.graph.edges():
                net.add_edge(edge[0], edge[1])
            net.set_options("""
            {
              "physics": {
                "hierarchicalRepulsion": {
                  "centralGravity": 0.0,
                  "springLength": 150,
                  "springConstant": 0.01,
                  "nodeDistance": 120
                },
                "solver": "hierarchicalRepulsion",
                "stabilization": {
                  "iterations": 1000
                }
              },
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "UD",
                  "sortMethod": "directed",
                  "levelSeparation": 150
                }
              }
            }
            """)
            net.write_html(output_file)
            print(f"Visualization saved to {output_file}")
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise e

    def save_graph_json(self, json_file):
        """
        Save detailed graph information to a JSON file.
        """
        try:
            graph_data = {
                "nodes": [],
                "edges": []
            }
            for node in self.graph.nodes():
                attrs = self.graph.nodes[node]
                graph_data["nodes"].append({
                    "id": node,
                    "level": attrs.get("level", 0),
                    "content": attrs.get("content", ""),
                    "font_size": attrs.get("font_size", None),
                    "is_underlined": attrs.get("is_underlined", False)
                })
            for edge in self.graph.edges():
                graph_data["edges"].append({
                    "from": edge[0],
                    "to": edge[1]
                })
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=4)
            print(f"Graph details saved to JSON file: {json_file}")
        except Exception as e:
            print(f"Error saving graph JSON: {e}")
            raise e

    def ocr_force(self, pdf_path):
        """
        Perform OCR on the PDF to extract text from images using Tesseract.
        This is a fallback method in case the regular text extraction fails or for image-based PDFs.
        """
        try:
            print(f"Starting OCR process for {pdf_path}")
            doc = fitz.open(pdf_path)
            extracted_text = []
            
            for page_num, page in enumerate(doc):
                # Get the pixmap (image) of the page for OCR processing.
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR on the image.
                text = pytesseract.image_to_string(img, lang="eng")
                
                # Add extracted text to the list with page information.
                extracted_text.append({
                    "text": text,
                    "page": page_num + 1
                })
            
            print(f"OCR extracted text from {len(extracted_text)} pages")
            return extracted_text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            raise e

    def ocr_force(self, doc):
        """
        Process a PDF document and apply OCR to pages with no text blocks.
        For pages that need OCR, creates a structured text dictionary.
        For pages with existing text, keeps the original page.
        Returns a list containing processed pages.
        """
        final_doc = []
        page_num = 0
        for page in doc:
            page_num += 1
            full_ocr_text = ""
            # Check if the page has text blocks
            need_ocr = False
            blocks = page.get_text("dict")["blocks"]
            if not blocks:  # Check if blocks is empty
                need_ocr = True
            if need_ocr:
                pixmap = page.get_pixmap()
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                try:
                    page_text = pytesseract.image_to_string(img)
                    full_ocr_text += f"==Start of OCR for page {page_num}==\n"
                    full_ocr_text += page_text
                    full_ocr_text += f"\n==End of OCR for page {page_num}==\n\n"
                    # print(f"Page {page_num} OCR text: {page_text}")
                    
                    # Create a structured representation of the OCR text
                    ocr_result = {
                        "text": page_text,
                        "font": "OCR-detected",
                        "font_size": 10.0,  # Default font size
                        "is_bold": False,   # We can't detect bold with basic OCR
                        "is_underlined": False,  # We can't detect underlines with basic OCR
                        "page": page_num,
                        "source": "ocr"     # Mark as OCR source for reference
                    }
                    final_doc.append(ocr_result)
                except Exception as e:
                    print(f"Error during OCR on page {page_num}: {e}")
                    # Add an empty result to keep page ordering consistent
                    final_doc.append({
                        "text": f"[OCR ERROR on page {page_num}]",
                        "font": "OCR-error",
                        "font_size": 10.0,
                        "is_bold": False,
                        "is_underlined": False,
                        "page": page_num,
                        "source": "ocr-error"
                    })
            else:
                final_doc.append(page)
        
        return final_doc

def main(pdf_path, output_html_path, output_json_path):
    try:
        print("Starting PDF to Knowledge Graph conversion (V5)")
        kg = PDFKnowledgeGraphV2()
        elements = kg.extract_structured_text_from_pdf(pdf_path)
        
        # If no elements were extracted, try OCR.
        if not elements:
            print("No text elements found, attempting OCR...")
            ocr_results = kg.ocr_force(pdf_path)
            # Convert OCR results to the expected format for graph building.
            elements = [{"text": res["text"], "is_bold": False, "is_underlined": False, "font_size": 8.0, "page": res["page"]} for res in ocr_results]
        
        kg.build_knowledge_graph(elements, pdf_path)
        kg.visualize(output_html_path)
        kg.save_graph_json(output_json_path)
        print("Knowledge graph conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pdf_to_kg_v2.py <pdf_path> <output_html_path> <output_json_path>")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
