import math

def get_direction(edge, tolerance = 0.001):
    x = edge[-1][0] - edge[0][0]
    y = edge[-1][1] - edge[0][1]
    
    if abs(x) < tolerance:
        x = 0
    if abs(y) < tolerance:
        y = 0
        
    if x != 0:
        x = x / abs(x)
    if y != 0:
        y = y / abs(y)
        
    if x !=0 and y != 0:
        print("not the edge in x or y")
    return [x, y]

def intersection(edge1, edge2, limit: bool = True, tolerance=0.0001):
    x1, y1 = edge1[0]
    x2, y2 = edge1[-1]
    x3, y3 = edge2[0]
    x4, y4 = edge2[-1]

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None  # Lines are parallel

    # Compute intersection point (infinite line)
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    # Check if the intersection is within both segments
    def in_range(p, a, b):
        return min(a, b) - tolerance <= p <= max(a, b) + tolerance  # allow small tolerance

    if not limit or (in_range(px, x1, x2) and in_range(py, y1, y2) and in_range(px, x3, x4) and in_range(py, y3, y4)):
        return [px, py]
    else:
        return None  # intersection is outside the segment bounds

def rotate(v, degree, tolerance = 0.0001):
    x = v[0]
    y = v[1]
    theta_rad = math.radians(-degree)  # Convert to radians
    x_new = x * math.cos(theta_rad) - y * math.sin(theta_rad)
    y_new = x * math.sin(theta_rad) + y * math.cos(theta_rad)
    if abs(x_new) < tolerance:
        x_new = 0
    if abs(y_new) < tolerance:
        y_new = 0
    return [x_new, y_new]

def distance(node1, node2):
    len = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
    return len

def nearby(node, edge):
    index = 0
    max_d = 1000000000
    for i, temp_node in enumerate(edge):
        temp_d = distance(temp_node, node)
        if temp_d < max_d:
            max_d = temp_d
            index = i
        else:
            break
    return index

def equal_node(node1, node2):
    tolerance = 0.001
    if node1[0] < node2[0] - tolerance or  node2[0] + tolerance < node1[0]:
        return False
    if node1[1] < node2[1] - tolerance or  node2[1] + tolerance < node1[1]:
        return False
    return True
        
def get_section(outline_list, expanding_list, pattern_lines_vertical: list, pattern_lines_horizon: list):   
    section_list = []
    
    for index, outline in enumerate(outline_list):
        ### center part LINE
        sub_section_list = []
        
        outline = outline_list[index]
        expanding = expanding_list[index]
        
        outline_pre = outline_list[index-1]
        expanding_pre = expanding_list[index-1]
        
        outline_post = outline_list[(index+1) % len(outline_list)]
        expanding_post = expanding_list[(index+1) % len(expanding_list)]
        
        outline_dir = get_direction(outline)
        if outline_dir[0] != 0:
            pattern_lines = pattern_lines_vertical
            pattern_lines_iteration = int(outline_dir[0]+outline_dir[1])
        else:
            pattern_lines = pattern_lines_horizon
            pattern_lines_iteration = int(outline_dir[0]+outline_dir[1])
            
        ### get the node1 & node2
        direction_pre_90 = rotate(get_direction(outline_pre), 90)
        direction = get_direction(outline)
        if direction == direction_pre_90:
            ### for the outer left corner
            section_node2 = [outline[0][0]+expanding[0], outline[0][1]+expanding[1]]
            section_node1 = outline[0]
        else:
            ### for the iner right corner
            section_node2 = [outline[0][0]+expanding[0]+expanding_pre[0], outline[0][1]+expanding[1]+expanding_pre[1]]
            section_node1 = outline[nearby(section_node2, outline)]
            for i in range(len(pattern_lines)-1, -1, pattern_lines_iteration):
                temp_node4 = outline[-1]
                temp_node3 = [outline[-1][0]+expanding[0], outline[-1][1]+expanding[1]]
                inner_intersection = intersection(pattern_lines[i], outline)
                outer_intersection = intersection(pattern_lines[i], [section_node2, temp_node3])
                if inner_intersection == section_node1:
                    section_node1 = outer_intersection
                    break
            
        ### get the node3 & node4 and build the corner section(right_corner_section)
        right_corner_section = None
        direction_post_270 = rotate(get_direction(outline_post), 270)
        direction = get_direction(outline)
        if direction == direction_post_270:
            ### for the outer right corner
            section_node3 = [outline[-1][0]+expanding[0], outline[-1][1]+expanding[1]]
            section_node4 = outline[-1]
            
            ### add the right section (outer)
            node1 = section_node4
            node2 = section_node3
            node3 = [outline[-1][0]+expanding[0]+expanding_post[0], outline[-1][1]+expanding[1]+expanding_post[1]]
            node4 = [outline[-1][0]+expanding_post[0], outline[-1][1]+expanding_post[1]]
            right_corner_section = {
                "type": "OUTER",
                "edge1": [node1, node2],
                "edge2": [node2, node3],
                "edge3": [node4, node3],
                "edge4": [node1, node4]
            }
        else:
            ### for the inner right corner
            section_node3 = [outline[-1][0]+expanding[0]+expanding_post[0], outline[-1][1]+expanding[1]+expanding_post[1]]
            section_node4 = outline[nearby(section_node3, outline)]

            right_corner_outer = section_node3
            
            outline_inner = outline
            outline_outer = [[outline[0][0]+expanding[0], outline[0][1]+expanding[1]], [outline[-1][0]+expanding[0], outline[-1][1]+expanding[1]]]
            for i in range(len(pattern_lines)-1, -1, pattern_lines_iteration):
                inner_intersection = intersection(pattern_lines[i], outline_inner)
                outer_intersection = intersection(pattern_lines[i], [section_node2, section_node3])
                if inner_intersection == section_node4:
                    section_node3 = outer_intersection
                    break
                
            ### find the right_corner_node3 and right_corner_node4
            next_section_node2 = right_corner_outer
            next_section_node1 = outline[nearby(next_section_node2, outline_post)]
            outline_post_dir = get_direction(outline_post)
            if outline_post_dir[0] != 0:
                pattern_post_lines = pattern_lines_vertical
                pattern_post_lines_iteration = int(outline_post_dir[0]+outline_post_dir[1])
            else:
                pattern_post_lines = pattern_lines_horizon
                pattern_post_lines_iteration = int(outline_post_dir[0]+outline_post_dir[1])
            outline_post_inner = outline_post
            outline_post_outer = [[outline_post[0][0]+expanding_post[0], outline_post[0][1]+expanding_post[1]], [outline_post[-1][0]+expanding_post[0], outline_post[-1][1]+expanding_post[1]]]
            for i in range(len(pattern_post_lines)-1, -1, pattern_post_lines_iteration):
                inner_intersection = intersection(pattern_lines[i], outline_post_inner)
                outer_intersection = intersection(pattern_lines[i], outline_post_outer)
                if inner_intersection == section_node4:
                    next_section_node2 = outer_intersection  
                    break
            
            if equal_node(right_corner_outer, section_node3) and equal_node(right_corner_outer, next_section_node2):
                right_corner_type = "INNER"
            elif equal_node(right_corner_outer, section_node3):
                right_corner_type = "INNER_RIGHT"
            elif equal_node(right_corner_outer, next_section_node2):
                right_corner_type = "INNER_LEFT"
            else:
                right_corner_type = "INNER_BOTH"
            left_index = nearby(section_node3, outline)
            right_index = nearby(next_section_node2, outline_post)
            right_corner_section = {
                "type": right_corner_type,
                "edge1": [outline[left_index], right_corner_outer],
                "edge2": [right_corner_outer, outline_post[right_index]],
                "edge3": outline_post[:right_index+1],
                "edge4": outline[left_index:],
                "outer": right_corner_outer
            }
                
        ### if the node1==node2, there is no "LINE"
        if not equal_node(section_node1, section_node4):
            outer_line = [section_node2, section_node3]
            
            edge4_begin_index = nearby(section_node1, outline)
            edge4_end_index = nearby(section_node4, outline)
            inner_line = outline[edge4_begin_index:edge4_end_index+1]

            ### loop to find each section of "LINE"
            post_index = 0
            begin_index = 0
            new_edge2 = [section_node2]
            left_edge = [section_node1, section_node2]
            for i in range(len(pattern_lines)-1, -1, pattern_lines_iteration):
                pattern_line = pattern_lines[i]
                inner_intersection = intersection(pattern_line, inner_line)
                outer_intersection = intersection(pattern_line, outer_line)
                
                if outer_intersection is not None:
                    new_edge2.append(outer_intersection)
                if inner_intersection is not None:
                    post_index = nearby(inner_intersection, inner_line)
                    if post_index == len(inner_line)-1:
                        break
                    
                    section_list.append({
                        "type": "LINE",
                        "edge1": left_edge,
                        "edge2": new_edge2,
                        "edge3": [inner_intersection, outer_intersection],
                        "edge4": inner_line[begin_index : post_index+1],
                    })
                    left_edge = [inner_intersection, outer_intersection]
                    new_edge2 = [new_edge2[-1]]
                    begin_index = post_index

            ### the last section of "LINE"
            if not equal_node(new_edge2[-1], section_node3):
                new_edge2.append(section_node3)
                section_node4_index = nearby(section_node4, left_edge)
                section_list.append({
                    "type": "LINE",
                    "edge1": left_edge,
                    "edge2": new_edge2,
                    "edge3": [section_node4, section_node3],
                    "edge4": inner_line[begin_index:],
                })
        
        section_list.append(right_corner_section)
    return section_list
            
def predict_section(criterias, element_size, left_num, index, section_list, last=2):
    ### get the predict
    if section["type"] == "OUTER":
        right_num_list = predict_wrap_outer(criterias, element_size, left_num, section)
    elif section["type"] == "INNER":
        right_num_list = predict_wrap_inner(criterias, element_size, left_num, section)
    elif section["type"] == "INNER_LEFT":
        right_num_list = predict_wrap_inner_left(criterias, element_size, left_num, section)
    elif section["type"] == "INNER_RIGHT":
        right_num_list = predict_wrap_inner_right(criterias, element_size, left_num, section)
    elif section["type"] == "INNER_BOTH":
        right_num_list = predict_wrap_inner_both(criterias, element_size, left_num, section)
    elif section["type"] == "LINE":
        right_num_list = predict_wrap_inner_line(criterias, element_size, left_num, section)
        
    ### loop right_num_list
    if index == len(section_list)-1:
        if last in right_num_list:
            section_list[index]["left_num"] = left_num
            section_list[index]["right_num"] = last
            return True
        else:
            return False
    for right_num in right_num_list:
        res = predict_section(criterias, element_size, right_num, index+1, section_list, last)
        if res:
            section_list[index]["left_num"] = left_num
            section_list[index]["right_num"] = right_num
            return True

    ### inpossible to build
    return False
            
def build_section(cdb_obj: 'Mesh2D', element_size, section_list):
    outline_list = []
    first_outline_left = None
    outline = []
    for index, section in enumerate(section_list):
        if section["type"] == "OUTER":
            outline_left, outline_right = build_wrap_outer(cdb_obj, element_size, section)
            
            if index == 0:
                first_outline_left = outline_left
            elif outline_left:
                outline_list.append(outline + outline_left[1:])
            outline = outline_right
        elif section["type"] == "INNER":
            outline_left, outline_right = build_wrap_inner(cdb_obj, element_size, section)
            
            if index == 0:
                first_outline_left = outline_left
            elif outline_left:
                outline_list.append(outline + outline_left[1:])
            outline = outline_right
        elif section["type"] == "INNER_LEFT":
            outline_left, outline_right = build_wrap_inner_left(cdb_obj, element_size, section)
            
            if index == 0:
                first_outline_left = outline_left
            elif outline_left:
                outline_list.append(outline + outline_left[1:])
            outline = outline_right
        elif section["type"] == "INNER_RIGHT":
            outline_left, outline_right = build_wrap_inner_right(cdb_obj, element_size, section)
                
            if index == 0:
                first_outline_left = outline_left
            elif outline_left:
                outline_list.append(outline + outline_left[1:])
            outline = outline_right
        elif section["type"] == "INNER_BOTH":
            outline_left, outline_right = build_wrap_inner_both(cdb_obj, element_size, section)
                
            if index == 0:
                first_outline_left = outline_left
            elif outline_left:
                outline_list.append(outline + outline_left[1:])
            outline = outline_right
        elif section["type"] == "LINE":
            outline_center = build_wrap_inner_line(cdb_obj, element_size, section)
            
            outline = outline + outline_center
    if first_outline_left is not None:
        if len(first_outline_left) > 0:
            outline_list.append(outline + first_outline_left[1:])
        else:
            outline_list.append(outline)
            
    return outline_list

def get_expanding(cdb_obj: 'Mesh2D', outline_list, element_size, pattern_lines_vertical: list, pattern_lines_horizon: list):
    return [element_size for _ in outline_list]

##################### 25/9/28 #######################
def get_expand_outline_simple(outline, expanding_list):
    node_list = []
    for i in range(len(outline)):
        pre = (i - 1 + len(outline)) % len(outline)
        x = outline[i][0][0] + expanding_list[pre][0] + expanding_list[i][0]
        y = outline[i][0][1] + expanding_list[pre][1] + expanding_list[i][1]
        node_list.append([x, y])
    new_outline = []
    for i in range(len(node_list)):
        post = (i + 1) % len(node_list)
        new_outline.append([node_list[i], node_list[post]])
    return new_outline

def get_new_pattern_line(outline, gap):
    ### find the gap area (<2*element_size)
    region_obj = Region()
    region_obj.set([edge[0] for edge in outline])
    region_obj.set_gap(gap)
    gap_polygon = region_obj.get_outline()
    merge_polygon = region_obj.get_outline(target_mask=1|2)
    
    ### identify the new edge and existed edge
    
    return new_pattern_vertical, new_pattern_horizon, auto_outline_list
        
    