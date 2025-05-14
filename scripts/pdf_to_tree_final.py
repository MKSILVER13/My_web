import networkx as nx
from pyvis.network import Network
import fitz  # PyMuPDF
import os
import re
import sys
import json  # new import for JSON output
import pytesseract
from PIL import Image

class PDFKnowledgeGraphV2:
    def __init__(self):
        self.graph = nx.DiGraph()

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
                final_doc.append(page) # Keep original page if no OCR needed
        return final_doc
        
    def extract_structured_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF with formatting information using PyMuPDF.
        Uses geometric analysis to detect underlined text.
        Returns a list of dictionaries with text and its formatting properties.
        """
        try:
            doc = fitz.open(pdf_path)
            processed_doc = self.ocr_force(doc) # Process with OCR first
            
            structured_text = []
            prev_font = None
            # Determine maximum font size from original document for reference
            max_font_size = 0
            for page_idx in range(len(doc)):
                page_content = doc.load_page(page_idx)
                blocks = page_content.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                max_font_size = max(max_font_size, span["size"])

            # Process each item from processed_doc (could be OCR dict or fitz.Page)
            for item in processed_doc:
                if isinstance(item, dict) and "source" in item and item["source"] in ["ocr", "ocr-error"]:
                    # This is an OCR result, add it directly
                    structured_text.append(item)
                    continue
                
                # This is a regular fitz.Page object, process it normally
                page = item # item is a fitz.Page here
                page_num = page.number # 0-indexed page number

                drawings = page.get_drawings()
                underline_rects = []
                for d in drawings:
                    if d.get("type") == "f":
                        for drawing_item_detail in d.get("items", []):
                            if drawing_item_detail[0] == "re":
                                rect = drawing_item_detail[1]
                                if rect.height < 5:
                                    underline_rects.append(rect)
                
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            is_bold = False
                            is_underlined = False
                            font_size = 0
                            font_used = ""
                            span_bboxes = []
                            
                            for span in line["spans"]:
                                if "bold" in span["font"].lower():
                                    is_bold = True
                                line_text += span["text"] + " "
                                font_size = round(max(font_size, span["size"]), ndigits=2)
                                if prev_font is None:
                                    prev_font = font_size
                                if prev_font-font_size < 0.2:
                                    font_size = prev_font
                                prev_font = round(max(font_size, span["size"]), ndigits=2)
                                font_used = span["font"]
                                span_bboxes.append(span["bbox"])
                            
                            if span_bboxes:
                                x0 = min(b[0] for b in span_bboxes)
                                y0 = min(b[1] for b in span_bboxes)
                                x1 = max(b[2] for b in span_bboxes)
                                y1 = max(b[3] for b in span_bboxes)
                                line_bbox = (x0, y0, x1, y1)
                            else:
                                line_bbox = None
                            
                            if line_bbox:
                                for rect in underline_rects:
                                    overlap = min(line_bbox[2], rect.x1) - max(line_bbox[0], rect.x0)
                                    if overlap > 0 and abs(rect.y0 - line_bbox[3]) < 5:
                                        is_underlined = True
                                        break
                            
                            line_text = line_text.strip()
                            if line_text:
                                structured_text.append({
                                    "text": line_text,
                                    "font": font_used,
                                    "font_size": font_size,
                                    "is_bold": is_bold,
                                    "is_underlined": is_underlined,
                                    "page": page_num + 1, # 1-indexed page number
                                    "source": "text" # Mark as normal text extraction
                                })
            return structured_text
        except Exception as e:
            raise e

    def build_knowledge_graph(self, elements, pdf_name):
        """
        Build a knowledge graph from the structured text elements.
        OCR text will be placed under a separate root node.
        """
        try:
            base_pdf_name = os.path.splitext(os.path.basename(pdf_name))[0]
            
            # Root node for normally extracted text
            root_normal_text = base_pdf_name
            self.graph.add_node(root_normal_text, level=0, content="", font_size=100, is_underlined=False, source="text_root")
            
            # Root node for OCR extracted text
            root_ocr_text = f"{base_pdf_name}_OCR"
            self.graph.add_node(root_ocr_text, level=0, content="OCR Content", font_size=100, is_underlined=False, source="ocr_root")

            # Stacks and current nodes for both trees
            node_stack_normal = [root_normal_text]
            current_node_normal = root_normal_text
            
            node_stack_ocr = [root_ocr_text]
            current_node_ocr = root_ocr_text
            
            first_bold_skipped_normal = False
            first_bold_skipped_ocr = False # Potentially skip first bold for OCR tree too
            
            # Determine regular font size (minimum of non-bold, non-OCR text)
            min_regular_font_size = float('inf')
            found_any_candidate_font = False
            for elem in elements:
                if elem.get("source") != "ocr" and elem.get("source") != "ocr-error" and not elem["is_bold"]:
                    min_regular_font_size = min(min_regular_font_size, elem["font_size"])
                    found_any_candidate_font = True
            
            regular_font_size = min_regular_font_size if found_any_candidate_font else 8.0
            # If you had a print statement here for debugging, it would show the chosen minimum:
            # print(f"Determined regular_font_size: {regular_font_size}")

            nodes_added = 0
            
            for i, elem in enumerate(elements):
                text = elem["text"].strip() # Strip text early
                is_bold = elem["is_bold"]
                is_underlined = elem["is_underlined"]
                font_size = elem["font_size"]
                source = elem.get("source", "text")

                # Determine current context (which stack, root, current_node, and skip_flag to use)
                active_node_stack = []
                current_parent_node_id = ""
                active_root_node_id = ""
                first_heading_skipped_for_stream = False # Local copy of the flag's current state

                if source == "ocr" or source == "ocr-error":
                    active_node_stack = node_stack_ocr
                    current_parent_node_id = current_node_ocr 
                    active_root_node_id = root_ocr_text
                    first_heading_skipped_for_stream = first_bold_skipped_ocr
                else:
                    active_node_stack = node_stack_normal
                    current_parent_node_id = current_node_normal
                    active_root_node_id = root_normal_text
                    first_heading_skipped_for_stream = first_bold_skipped_normal
                
                # Calculate next_starts_with_colon
                next_starts_with_colon = False
                if i < len(elements) - 1:
                    next_elem = elements[i+1]
                    if next_elem.get("source", "text") == source:
                        # Original logic for colon check based on boldness for normal text
                        if not (source == "ocr" or source == "ocr-error") and not next_elem["is_bold"]:
                            next_starts_with_colon = next_elem["text"].startswith(":")
                        # For OCR, original logic didn't check bold status of next element for colon
                        elif (source == "ocr" or source == "ocr-error"): 
                             next_starts_with_colon = next_elem["text"].startswith(":")
                
                # Determine if the current element is a potential heading
                is_potential_heading = False
                if source == "ocr" or source == "ocr-error":
                    is_potential_heading = text.endswith(':') or next_starts_with_colon or (font_size > 10.0)
                else: # Normal text
                    if font_size > regular_font_size: is_potential_heading = True
                    elif is_bold: is_potential_heading = True 
                    elif is_underlined: is_potential_heading = True 
                    elif text.endswith(':') or next_starts_with_colon: is_potential_heading = True

                if re.match(r'^\\d+\\.\\s*$', text): # Skip standalone numbered items
                    # Update stream's current node to itself as no new node is made
                    if source == "ocr" or source == "ocr-error": current_node_ocr = current_parent_node_id
                    else: current_node_normal = current_parent_node_id
                    continue

                new_node_was_created = False
                new_parent_for_next_iteration = current_parent_node_id # Default to existing parent

                if is_potential_heading:
                    if not first_heading_skipped_for_stream: # If it's the first heading for this stream
                        # Update the persistent skip flag for the stream
                        if source == "ocr" or source == "ocr-error": first_bold_skipped_ocr = True
                        else: first_bold_skipped_normal = True
                        
                        # Add title text to the active_root_node_id's content
                        if active_root_node_id in self.graph.nodes:
                            self.graph.nodes[active_root_node_id]["content"] += " " + text
                        # No new node is created; new_parent_for_next_iteration remains current_parent_node_id (which is likely the root here)
                    else:
                        # This is a subsequent heading, create a node and place it hierarchically
                        # Determine parent using a temporary copy of the stack for traversal
                        temp_stack_for_placement = list(active_node_stack)
                        determined_parent_for_new_node = active_root_node_id # Default to root

                        while len(temp_stack_for_placement) > 0:
                            candidate_on_stack = temp_stack_for_placement[-1]
                            if candidate_on_stack == active_root_node_id: # Cannot go above root via stack
                                determined_parent_for_new_node = active_root_node_id
                                break

                            candidate_attrs = self.graph.nodes[candidate_on_stack]
                            cand_fs = candidate_attrs["font_size"]
                            print(cand_fs)
                            cand_bold = candidate_attrs.get("is_bold", False)
                            cand_underlined = candidate_attrs.get("is_underlined", False)

                            if font_size > cand_fs: temp_stack_for_placement.pop(); continue
                            if font_size < cand_fs: determined_parent_for_new_node = candidate_on_stack; break
                            # font_size == cand_fs
                            if is_bold and not cand_bold: temp_stack_for_placement.pop(); continue
                            if not is_bold and cand_bold: determined_parent_for_new_node = candidate_on_stack; break
                            # is_bold == cand_bold
                            if is_underlined and not cand_underlined: temp_stack_for_placement.pop(); continue
                            if not is_underlined and cand_underlined: determined_parent_for_new_node = candidate_on_stack; break
                            # is_underlined == cand_underlined (siblings)
                            temp_stack_for_placement.pop() # Pop candidate to attach to its parent
                            determined_parent_for_new_node = temp_stack_for_placement[-1] if temp_stack_for_placement else active_root_node_id
                            break 
                        
                        if not temp_stack_for_placement : # If stack became empty, parent is root
                            determined_parent_for_new_node = active_root_node_id

                        # Adjust the actual active_node_stack to reflect the determined parent
                        while len(active_node_stack) > 0 and active_node_stack[-1] != determined_parent_for_new_node:
                            if active_node_stack[-1] == active_root_node_id : break # Stop if root is reached and it's not the parent
                            active_node_stack.pop()
                        
                        # Ensure stack is not empty if parent is root
                        if not active_node_stack and determined_parent_for_new_node == active_root_node_id:
                            active_node_stack.append(active_root_node_id)
                        elif len(active_node_stack) > 0 and active_node_stack[-1] != determined_parent_for_new_node :
                             # Fallback: if stack is messed up, reset to root
                            active_node_stack.clear()
                            active_node_stack.append(active_root_node_id)
                            determined_parent_for_new_node = active_root_node_id


                        new_node_id_val = f"{text} ({source} p{elem['page']})"
                        _counter = 0
                        _original_new_node_id = new_node_id_val
                        while self.graph.has_node(new_node_id_val):
                            _counter += 1
                            new_node_id_val = f"{_original_new_node_id}_{_counter}"

                        self.graph.add_node(new_node_id_val, level=len(active_node_stack), content="", font_size=font_size, is_bold=is_bold, is_underlined=is_underlined, source=source, page=elem['page'])
                        self.graph.add_edge(determined_parent_for_new_node, new_node_id_val)
                        
                        active_node_stack.append(new_node_id_val)
                        new_parent_for_next_iteration = new_node_id_val
                        nodes_added += 1
                        new_node_was_created = True
                
                if not new_node_was_created and not (is_potential_heading and not first_heading_skipped_for_stream) : 
                    # This is content for current_parent_node_id
                    # (Avoids adding skipped title's text twice if it was already added to root)
                    if current_parent_node_id in self.graph.nodes:
                         self.graph.nodes[current_parent_node_id]["content"] += " " + text
                
                # Update the main current_node trackers for the respective stream
                if source == "ocr" or source == "ocr-error":
                    current_node_ocr = new_parent_for_next_iteration
                else:
                    current_node_normal = new_parent_for_next_iteration
            
            # Create a node for each page in case of OCR (This logic seems separate and can be kept as is)
            ocr_page_nodes = {}
            for elem in elements:
                if elem.get("source") == "ocr":
                    page_node_id = f"Page {elem['page']} OCR"
                    if page_node_id not in ocr_page_nodes:
                        self.graph.add_node(page_node_id, level=1, content="", font_size=10.0, is_underlined=False, source="ocr-page", page=elem['page'])
                        self.graph.add_edge(root_ocr_text, page_node_id)
                        ocr_page_nodes[page_node_id] = True

            # Link OCR nodes to their respective page nodes
            for elem in elements:
                if elem.get("source") == "ocr":
                    page_node_id = f"Page {elem['page']} OCR"
                    if page_node_id in ocr_page_nodes:
                        new_node_id = f"{elem['text']} (OCR p{elem['page']})"
                        self.graph.add_node(new_node_id, level=2, content=elem['text'], font_size=elem['font_size'], is_underlined=elem['is_underlined'], source="ocr", page=elem['page'])
                        self.graph.add_edge(page_node_id, new_node_id)
            
            for node_id in list(self.graph.nodes()): # Iterate over a copy for safe modification
                if "content" in self.graph.nodes[node_id]:
                    content = self.graph.nodes[node_id]["content"].strip()
                    self.graph.nodes[node_id]["content"] = content if content else "(No description)"
            
            # --- Leaf Node Merging for Normal Text Tree ---
            parents_and_their_leaf_children_to_merge = []
            
            candidate_parent_ids = [
                n for n in self.graph.nodes() 
                if self.graph.nodes[n].get("source") not in ["ocr", "ocr-page", "ocr_root", "ocr-error"]
            ]

            for parent_id in candidate_parent_ids:
                children_ids = list(self.graph.successors(parent_id))
                
                current_parent_normal_text_leaf_children = []
                for child_id in children_ids:
                    if not self.graph.has_node(child_id): # Safety check
                        continue

                    child_attrs = self.graph.nodes[child_id]
                    child_source = child_attrs.get("source")

                    # Skip OCR-related children
                    if child_source in ["ocr", "ocr-page", "ocr_root", "ocr-error"]:
                        continue 
                    
                    # Check if this non-OCR child is a leaf node
                    if self.graph.out_degree(child_id) == 0:
                        current_parent_normal_text_leaf_children.append(child_id)
                
                # If this parent has more than one normal text leaf child, they are candidates for merging
                if len(current_parent_normal_text_leaf_children) > 1:
                    parents_and_their_leaf_children_to_merge.append((parent_id, current_parent_normal_text_leaf_children))

            merged_node_operations_count = 0
            for parent_id, leaves_to_merge_list in parents_and_their_leaf_children_to_merge:
                # Ensure parent still exists and is not an OCR node
                if not self.graph.has_node(parent_id) or \
                   self.graph.nodes[parent_id].get("source") in ["ocr", "ocr-page", "ocr_root", "ocr-error"]:
                    continue
                
                # Filter leaves_to_merge_list to only those that still exist.
                # This is a safety check, as a node might have been removed if it was part of another merge operation
                # (though less likely with the current iteration structure, it's good practice).
                actual_leaves_to_merge_now = [
                    leaf_id for leaf_id in leaves_to_merge_list 
                    if self.graph.has_node(leaf_id) and \
                       self.graph.has_edge(parent_id, leaf_id) and \
                       self.graph.nodes[leaf_id].get("source") not in ["ocr", "ocr-page", "ocr_root", "ocr-error"] and \
                       self.graph.out_degree(leaf_id) == 0
                ]
                
                if len(actual_leaves_to_merge_now) <= 1: # Not enough leaves to merge after filtering
                    continue

                merged_content = ""
                max_font_size = 0.0
                any_bold = False
                any_underlined = False
                first_leaf_page = None
                current_leaf_level = None 

                for leaf_id in actual_leaves_to_merge_now:
                    leaf_attrs = self.graph.nodes[leaf_id]
                    merged_content += leaf_attrs.get("content", "").strip() + " " 
                    max_font_size = max(max_font_size, leaf_attrs.get("font_size", 0.0))
                    if leaf_attrs.get("is_bold", False): any_bold = True
                    if leaf_attrs.get("is_underlined", False): any_underlined = True
                    
                    if first_leaf_page is None: first_leaf_page = leaf_attrs.get("page")
                    if current_leaf_level is None: current_leaf_level = leaf_attrs.get("level")

                if current_leaf_level is None: # Fallback if level couldn't be determined
                    parent_level = self.graph.nodes[parent_id].get("level", 0)
                    current_leaf_level = parent_level + 1
                
                merged_node_id_base = f"{parent_id}_merged_cluster"
                merged_node_id = merged_node_id_base
                _temp_counter = 0
                while self.graph.has_node(merged_node_id):
                    _temp_counter += 1
                    merged_node_id = f"{merged_node_id_base}_{_temp_counter}"
                
                self.graph.add_node(
                    merged_node_id,
                    level=current_leaf_level,
                    content=merged_content.strip(),
                    font_size=max_font_size,
                    is_bold=any_bold,
                    is_underlined=any_underlined,
                    source="text", 
                    page=first_leaf_page
                )
                self.graph.add_edge(parent_id, merged_node_id)
                
                for leaf_id in actual_leaves_to_merge_now:
                    if self.graph.has_node(leaf_id): 
                        self.graph.remove_node(leaf_id)
                
                merged_node_operations_count += 1
            
            if merged_node_operations_count > 0:
                print(f"Performed {merged_node_operations_count} leaf node merging operations in the normal text tree.")

            # Final content stripping
            for node_id in self.graph.nodes():
                if "content" in self.graph.nodes[node_id]:
                    content = self.graph.nodes[node_id]["content"].strip()
                    self.graph.nodes[node_id]["content"] = content if content else "(No description)"
        except Exception as e:
            print(f"Error building knowledge graph: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def visualize(self, output_file):
        """
        Visualize the knowledge graph using PyVis with color coding:
          - Different colors for different levels.
          - Underlined nodes get a special gold color.
          - OCR nodes get a distinct color (e.g., light blue).
        """
        try:
            net = Network(height="750px", width="100%", directed=True)
            colors = {
                0: '#FF6347',  # Root - Tomato Red
                1: '#4682B4',  # Level 1 - Steel Blue
                2: '#32CD32',  # Level 2 - Lime Green
                3: '#FFD700',  # Level 3 - Gold
                4: '#6A5ACD',  # Level 4 - Slate Blue
                5: '#FF69B4',  # Level 5 - Hot Pink
                6: '#00CED1',  # Level 6 - Dark Turquoise
            }
            ocr_color = '#AFEEEE'  # Pale Turquoise for OCR nodes
            ocr_root_color = '#20B2AA' # Light Sea Green for OCR root
            ocr_page_color = '#B0E0E6' # Powder Blue for OCR Page nodes

            for node in self.graph.nodes():
                attrs = self.graph.nodes[node]
                level = attrs.get("level", 0)
                node_source = attrs.get("source", "text")
                
                color = colors.get(level, "#D3D3D3") # Default color - Light Grey
                
                if node_source == "text_root":
                    color = colors.get(0) # Ensure text_root gets the root color
                elif node_source == "ocr_root":
                    color = ocr_root_color
                elif node_source == "ocr-page":
                    color = ocr_page_color
                elif node_source == "ocr" or node_source == "ocr-error":
                    color = ocr_color
                elif attrs.get("is_underlined", False):
                    color = "#DAA520"  # Goldenrod for underlined nodes (takes precedence over level color for non-OCR)
                
                label = node if len(node) <= 40 else node[:37] + "..." # Increased label length slightly
                content = attrs.get("content", "")
                title = f"{node}\nPage: {attrs.get('page', 'N/A')}\nSource: {node_source}\nFont Size: {attrs.get('font_size', 'N/A')}\nUnderlined: {attrs.get('is_underlined', False)}\n\n{content}" if content else f"{node}\nPage: {attrs.get('page', 'N/A')}\nSource: {node_source}"
                net.add_node(node, label=label, title=title, color=color)
            
            for edge in self.graph.edges():
                net.add_edge(edge[0], edge[1])
            
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "hierarchicalRepulsion": {
                  "centralGravity": 0.0,
                  "springLength": 200,
                  "springConstant": 0.01,
                  "nodeDistance": 150,
                  "damping": 0.09
                },
                "minVelocity": 0.75,
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
                  "levelSeparation": 150,
                  "treeSpacing": 200,
                  "blockShifting": true,
                  "edgeMinimization": true,
                  "parentCentralization": true
                }
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 200
              },
              "nodes": {
                "shape": "box",
                "font": {
                    "size": 12
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

def main(pdf_path, output_html_path, output_json_path):
    try:
        print("Starting PDF to Knowledge Graph conversion (V5)")
        kg = PDFKnowledgeGraphV2()
        elements = kg.extract_structured_text_from_pdf(pdf_path)
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
