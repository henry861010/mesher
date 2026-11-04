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
        
def adapter_get_section(outline_list, expanding_list, pattern_lines_vertical: list, pattern_lines_horizon: list):   
    section_list = []
    right_corner_section = None
    
    for index, outline in enumerate(outline_list):
        if len(outline_list[index]) < 2:
            raise ValueError(f"[adapter_get_section] the len of outline {len(outline_list[index])} is small than 2")
            
        outline = outline_list[index]
        expanding = expanding_list[index]
        direction = get_direction(outline)
        
        outline_pre = outline_list[index-1]
        expanding_pre = expanding_list[index-1]
        direction_pre = get_direction(outline_pre)
        
        outline_post = outline_list[int(index+1)%len(outline_list)]
        expanding_post = expanding_list[int(index+1)%len(outline_list)]
        direction_post = get_direction(outline_post)
        
        outer_line = [[outline[0][0]+expanding[0], outline[0][1]+expanding[1]], [outline[-1][0]+expanding[0], outline[-1][1]+expanding[1]]]
        outer_line_post = [[outline_post[0][0]+expanding_post[0], outline_post[0][1]+expanding_post[1]], [outline_post[-1][0]+expanding_post[0], outline_post[-1][1]+expanding_post[1]]]
        
        if direction[0] != 0:
            if direction[0] < 0:
                pattern_line_list = list(reversed(pattern_lines_vertical))
            else:
                pattern_line_list = pattern_lines_vertical
        else:
            if direction[1] < 0:
                pattern_line_list = list(reversed(pattern_lines_horizon))
            else:
                pattern_line_list = pattern_lines_horizon

        if direction_post[0] != 0:
            if direction_post[0] < 0:
                pattern_line_post_list = list(reversed(pattern_lines_vertical))
            else:
                pattern_line_post_list = pattern_lines_vertical
        else:
            if direction_post[1] < 0:
                pattern_line_post_list = list(reversed(pattern_lines_horizon))
            else:
                pattern_line_post_list = pattern_lines_horizon
        
        ### [section_node1 & section_node2]
        if direction == rotate(direction_pre, 90):
            ### left - outer corner
            section_node1 = outline[0]
            section_node2 = [outline[0][0]+expanding[0], outline[0][1]+expanding[1]]
            begin_index = 0
        else:
            ### left - inner corner
            section_node2 = [outline[0][0]+expanding[0]+expanding_pre[0], outline[0][1]+expanding[1]+expanding_pre[1]]
            begin_index = max(1, nearby(section_node2, outline))
            section_node1 = outline[begin_index]
            
            ### check for inner-right corner
            for pattern_line in pattern_line_list:
                inner_intersection = intersection(pattern_line, outline)
                if inner_intersection is not None:
                    if equal_node(inner_intersection, section_node1):
                        section_node2 = intersection(pattern_line, outer_line, limit=False)
                        begin_index = max(1, nearby(section_node2, outline))
                        section_node1 = outline[begin_index]
                    break

            
        ### [section_node3 & section_node4]
        if direction == rotate(direction_post, 270):
            ### right outer corner
            section_node4 = outline[-1]
            section_node3 = [outline[-1][0]+expanding[0], outline[-1][1]+expanding[1]]
            end_index = len(outline)-1
            
            ### add right corner section
            corner_node1 = section_node4
            corner_node2 = section_node3
            corner_node3 = [outline[-1][0]+expanding[0]+expanding_post[0], outline[-1][1]+expanding[1]+expanding_post[1]]
            corner_node4 = [outline[-1][0]+expanding_post[0], outline[-1][1]+expanding_post[1]]
            right_corner_section = {
                "type": "OUTER",
                "edge1": [corner_node1, corner_node2],
                "edge2": [corner_node2, corner_node3],
                "edge3": [corner_node4, corner_node3],
                "edge4": [corner_node1, corner_node4]
            }
        else:
            ### right inner corner
            right_outer = [outline[-1][0]+expanding[0]+expanding_post[0], outline[-1][1]+expanding[1]+expanding_post[1]]
            
            ### right inner corner - left side            
            section_node3 = right_outer
            end_index = min(len(outline)-2, nearby(section_node3, outline))
            section_node4 = outline[end_index]
            
            ### check for inner-left corner
            for pattern_line in list(reversed(pattern_line_list)):
                inner_intersection = intersection(pattern_line, outline)
                if inner_intersection is not None:
                    if equal_node(section_node4, inner_intersection):
                        section_node3 = intersection(pattern_line, outer_line, limit=False)
                        end_index = min(len(outline)-2, nearby(section_node3, outline))
                        section_node4 = outline[end_index]
                    break
            
            ### right inner corner - right side (next_section_node1 & next_section_node2)
            next_section_node2 = right_outer  
            next_begin_index = max(1, nearby(next_section_node2, outline_post))
            next_section_node1 = outline_post[next_begin_index]
            
            ### check for inner-right corner
            for pattern_line_post in pattern_line_post_list:
                inner_intersection = intersection(pattern_line_post, outline_post)
                if inner_intersection is not None:
                    if equal_node(inner_intersection, next_section_node1):
                        next_section_node2 = intersection(pattern_line_post, outer_line_post, limit=False)
                        next_begin_index = max(1, nearby(next_section_node2, outline_post))
                        next_section_node1 = outline_post[next_begin_index]
                    break
            
            ### add right corner section
            if not equal_node(right_outer, section_node3) and not equal_node(right_outer, next_section_node2):
                right_corner_section = {
                    "type": "INNER_BOTH",
                    "edge1": [section_node4, section_node3],
                    "edge2": [section_node3, right_outer],
                    "edge3": [right_outer, next_section_node2],
                    "edge4": [next_section_node1, next_section_node2],
                    "edge5": outline[end_index:],
                    "edge6": outline_post[:next_begin_index+1]
                }
            elif equal_node(right_outer, section_node3) and not equal_node(right_outer, next_section_node2):
                right_corner_section = {
                    "type": "INNER_RIGHT",
                    "edge1": [section_node4, right_outer, next_section_node2],
                    "edge2": [next_section_node2, next_section_node1],
                    "edge3": outline_post[:next_begin_index+1],
                    "edge4": outline[end_index:]
                }
            elif not equal_node(right_outer, section_node3) and equal_node(right_outer, next_section_node2):
                right_corner_section = {
                    "type": "INNER_LEFT",
                    "edge1": [section_node4, right_outer],
                    "edge2": [section_node3, right_outer, next_section_node1],
                    "edge3": outline_post[:next_begin_index+1],
                    "edge4": outline[end_index:]
                }
            else:
                right_corner_section = {
                    "type": "INNER",
                    "edge1": [section_node4, right_outer],
                    "edge2": [right_outer, next_section_node1],
                    "edge3": outline_post[:next_begin_index+1],
                    "edge4": outline[end_index:]
                }

        ### [center line ]if section_node2 != section_node2
        if equal_node(section_node1, section_node4):
            raise ValueError(f"[adapter_get_section] section_node1 and section_node4 are overlap, out of consideration. {section_node1} & {section_node4}")
        if equal_node(section_node2, section_node3):
            raise ValueError(f"[adapter_get_section] section_node2 and section_node3 are overlap, out of consideration. {section_node2} & {section_node3}")
        
        inner_line = outline[begin_index:end_index+1]
        
        left_edge = [section_node1, section_node2]
        new_edge2 = [section_node2]
        pre_index = 0
        post_index = 0
        
        ### begin
        for pattern_line in pattern_line_list:
            inner_intersection = intersection(pattern_line, outline)
            outer_intersection = intersection(pattern_line, outer_line)
            if outer_intersection is not None:
                if not equal_node(new_edge2[-1], outer_intersection):
                    new_edge2.append(outer_intersection)
            if inner_intersection is not None:
                if not equal_node(inner_intersection, left_edge[0]):
                    ### append the outer node
                    outer_intersection = intersection(pattern_line, outer_line, limit=False)
                    if not equal_node(new_edge2[-1], outer_intersection):
                        new_edge2.append(outer_intersection)
                    
                    post_index = nearby(inner_intersection, inner_line)
                    
                    if not equal_node(inner_intersection, inner_line[post_index]):
                        raise ValueError(f"[adapter_get_section] there should be node at edge4 for LINE which equal to inner_intersection, but not equal. {inner_intersection}-{inner_line[post_index]}")
                    
                    ### append the section
                    section_list.append({
                        "type": "LINE",
                        "edge1": left_edge,
                        "edge2": new_edge2,
                        "edge3": [inner_intersection, outer_intersection],
                        "edge4": inner_line[pre_index:post_index+1]
                    })
                    
                    pre_index = post_index
                    left_edge = [inner_intersection, outer_intersection]
                    new_edge2 = [new_edge2[-1]]
                
        ### last
        if pre_index != len(inner_line)-1:
            if not equal_node(new_edge2[-1], section_node3):
                new_edge2.append(section_node3)
            section_list.append({
                "type": "LINE",
                "edge1": left_edge,
                "edge2": new_edge2,
                "edge3": [section_node4, section_node3],
                "edge4": inner_line[pre_index:]
            })
        
        ### add the right corner section
        section_list.append(right_corner_section)
    return section_list