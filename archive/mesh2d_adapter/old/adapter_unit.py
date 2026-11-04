def build_unit_1_1_1_x(cdb_obj: 'Mesh2D', edge1: list, edge2: list, edge3: list, edge4: list):    
    origin_edge1 = edge1
    origin_edge2 = edge2
    origin_edge3 = edge3
    origin_edge4 = edge4
    if len(edge1) > 2:
        edge1 = origin_edge4
        edge2 = origin_edge3
        edge3 = origin_edge2
        edge4 = origin_edge1
    elif len(edge2) > 2:
        edge1 = origin_edge3[::-1]
        edge2 = origin_edge4[::-1]
        edge3 = origin_edge1[::-1]
        edge4 = origin_edge2[::-1]
    elif len(edge3) > 2:
        edge1 = origin_edge4[::-1]
        edge2 = origin_edge1
        edge3 = origin_edge2[::-1]
        edge4 = origin_edge3
    else:
        edge1 = origin_edge1
        edge2 = origin_edge2
        edge3 = origin_edge3
        edge4 = origin_edge4  
        
    ### case 1-1-1-1
    if len(edge1)==2 and len(edge2)==2 and len(edge3)==2 and len(edge4)==2:
        cdb_obj.add_element(edge1[0], edge2[0], edge3[-1], edge4[-1])
        return
        
    if not (len(edge1)==2 and len(edge2) and len(edge3)):
        raise ValueError(f"[build_unit_1_1_1_x] the format of the edges is not 1-1-1-n, edge1:{len(origin_edge1)}, edge2:{len(origin_edge2)}, edge3:{len(origin_edge3)}, edge4:{len(origin_edge4)}")
    if (len(edge1) + len(edge2) + len(edge3) + len(edge4)) % 2 == 1:
        raise ValueError(f"[build_unit_1_1_1_x] the sum of edges is not Even, edge1:{len(origin_edge1)}, edge2:{len(origin_edge2)}, edge3:{len(origin_edge3)}, edge4:{len(origin_edge4)}")
        
    node1 = edge1[0]
    node2 = edge2[0]
    node3 = edge3[-1]
    node4 = edge4[-1]
    
    section_num = len(edge4) - 1
    half_num = section_num // 2
    
    if section_num % 2 == 0:
        raise Exception("[build_unit_normal] the number of nodes at edge4 should be even")
    
    ### vector1
    vector1 = edge4[0 : half_num+1]
    vector1.reverse()
    
    ### vector4
    vector4 = edge4[half_num+1:]
    
    ### vector2
    mid1_x = node1[0] + (node4[0] - node1[0]) * half_num / section_num
    mid1_y = node1[1] + (node4[1] - node1[1]) * half_num / section_num
    mid2_x = node2[0] + (node3[0] - node2[0]) * half_num / section_num
    mid2_y = node2[1] + (node3[1] - node2[1]) * half_num / section_num
    mid_x = (mid1_x + mid2_x) / 2
    mid_y = (mid1_y + mid2_y) / 2
    temp_x = mid_x
    temp_y = mid_y
    v_x = (node2[0] - mid_x) / half_num
    v_y = (node2[1] - mid_y) / half_num
    vector2 = []
    for _ in range(half_num+1):
        vector2.append([temp_x, temp_y])
        temp_x += v_x
        temp_y += v_y

    ### vector3
    mid1_x = node1[0] + (node4[0] - node1[0]) * (half_num + 1) / section_num
    mid1_y = node1[1] + (node4[1] - node1[1]) * (half_num + 1) / section_num
    mid2_x = node2[0] + (node3[0] - node2[0]) * (half_num + 1) / section_num
    mid2_y = node2[1] + (node3[1] - node2[1]) * (half_num + 1) / section_num
    mid_x = (mid1_x + mid2_x) / 2
    mid_y = (mid1_y + mid2_y) / 2
    temp_x = mid_x
    temp_y = mid_y
    v_x = (node3[0] - mid_x) / half_num
    v_y = (node3[1] - mid_y) / half_num
    vector3 = []
    for _ in range(half_num+1):
        vector3.append([temp_x, temp_y])
        temp_x += v_x
        temp_y += v_y
    
    ### build the element
    ### center_node
    cdb_obj.add_element(vector1[0], vector2[0], vector3[0], vector4[0])
    for i in range(1, half_num+1):
        ### left
        cdb_obj.add_element(vector1[i-1], vector1[i], vector2[i], vector2[i-1])
        
        ### center
        cdb_obj.add_element(vector2[i-1], vector2[i], vector3[i], vector3[i-1])
        
        ### right
        cdb_obj.add_element(vector3[i-1], vector3[i], vector4[i], vector4[i-1])
        
    return [origin_edge1, origin_edge2, origin_edge3, origin_edge4]

def build_unit_x_1_1_y(cdb_obj: 'Mesh2D', edge1: list, edge2: list, edge3: list, edge4: list):
    origin_edge1 = edge1
    origin_edge2 = edge2
    origin_edge3 = edge3
    origin_edge4 = edge4
    if len(edge1) > 2 and len(edge2) > 2:
        edge1 = origin_edge2
        edge2 = origin_edge3[::-1]
        edge3 = origin_edge4
        edge4 = origin_edge1[::-1]
    elif len(edge2) > 2 and len(edge3) > 2:
        edge1 = origin_edge3[::-1]
        edge2 = origin_edge4[::-1]
        edge3 = origin_edge1[::-1]
        edge4 = origin_edge2[::-1]
    elif len(edge3) > 2 and len(edge4) > 2:
        edge1 = origin_edge4[::-1]
        edge2 = origin_edge1
        edge3 = origin_edge2[::-1]
        edge4 = origin_edge3
    else:
        edge1 = origin_edge1
        edge2 = origin_edge2
        edge3 = origin_edge3
        edge4 = origin_edge4  
        
    ### case 1-1-1-1
    if len(edge1)==2 and len(edge2)==2 and len(edge3)==2 and len(edge4)==2:
        cdb_obj.add_element(edge1[0], edge2[0], edge3[-1], edge4[-1])
        return  
        
    if len(edge2)!=2 or len(edge3)!=2:
        raise ValueError(f"[build_unit_x_1_1_y] the format of the edges is not x-1-1-y, edge1:{len(origin_edge1)}, edge2:{len(origin_edge2)}, edge3:{len(origin_edge3)}, edge4:{len(origin_edge4)}")
    if (lend(edge1) + len(edge2) + len(edge3) + len(edge4)) % 2 == 1:
        raise ValueError(f"[build_unit_x_1_1_y] the sum of edges is not Even, edge1:{len(origin_edge1)}, edge2:{len(origin_edge2)}, edge3:{len(origin_edge3)}, edge4:{len(origin_edge4)}")
        
    node1 = edge1[0]
    node2 = edge2[0]
    node3 = edge3[-1]
    node4 = edge4[-1]
    
    if len(edge1) > len(edge4):
        l_edge = edge1
        s_edge = edge4
        lr_edge = edge3
        sr_edge = edge2
    else:
        l_edge = edge4
        s_edge = edge1
        lr_edge = edge2
        sr_edge = edge3
    
    l_begin = 1 + len(l_edge) - len(s_edge)
    s_begin = 1
    outer_section_num = len(s_edge) - 2
    
    ### mid point
    if len(s_edge) == 1:
        mid = lr_edge[-1]
    else:
        temp1 = l_edge[l_begin]
        temp3_x = lr_edge[0][0] + (distance(l_edge[l_begin], l_edge[0])/distance(l_edge[-1], l_edge[0])) * (lr_edge[-1][0] - lr_edge[0][0])
        temp3_y = lr_edge[0][1] + (distance(l_edge[l_begin], l_edge[0])/distance(l_edge[-1], l_edge[0])) * (lr_edge[-1][1] - lr_edge[0][1])
        temp3 = [temp3_x, temp3_y] 
        temp2_x = sr_edge[0][0] + (distance(s_edge[s_begin], s_edge[0])/distance(s_edge[-1], s_edge[0])) * (sr_edge[-1][0] - sr_edge[0][0])
        temp2_y = sr_edge[0][1] + (distance(s_edge[s_begin], s_edge[0])/distance(s_edge[-1], s_edge[0])) * (sr_edge[-1][1] - sr_edge[0][1])
        temp2 = [temp2_x, temp2_y] 
        temp4 = s_edge[s_begin]
        temp_edge1 = [[temp1[0], temp1[1]], [temp3[0], temp3[1]]]
        temp_edge2 = [[temp4[0], temp4[1]], [temp2[0], temp2[1]]]
        mid = intersection(temp_edge1, temp_edge2)
    
    ### build the inner part
    temp_edge1 = l_edge[0:l_begin+1]
    temp_edge2 = [l_edge[l_begin], mid]
    temp_edge3 = [s_edge[s_begin], mid]
    temp_edge4 = s_edge[0:s_begin+1]
    build_unit_normal(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)
    
    ### build the other layer - build vector
    if outer_section_num > 0:
        vector1 = l_edge[l_begin:]
        vector3 = s_edge[s_begin:]
        temp_x = mid[0]
        temp_y = mid[1]
        v_x = (node3[0] - mid[0]) / outer_section_num
        v_y = (node3[1] - mid[1]) / outer_section_num
        vector2 = []
        for _ in range(outer_section_num+1):
            vector2.append([temp_x, temp_y])
            temp_x += v_x
            temp_y += v_y
            
        ### build the other layer - build element
        for i in range(len(vector1)):
            ### right
            cdb_obj.add_element(vector1[i-1], vector1[i], vector2[i], vector2[i-1])
            
            ### left
            cdb_obj.add_element(vector2[i-1], vector2[i], vector3[i], vector3[i-1])    
    return [origin_edge1, origin_edge2, origin_edge3, origin_edge4]    

def build_unit_x_y_z_1(cdb_obj: 'Mesh2D', edge1: list, edge2: list, edge3: list, edge4: list):
    origin_edge1 = edge1
    origin_edge2 = edge2
    origin_edge3 = edge3
    origin_edge4 = edge4
    if len(edge1) ==2:
        edge4 = origin_edge1[::-1]
        edge1 = origin_edge2
        edge2 = origin_edge3[::-1]
        edge3 = origin_edge4
    elif len(edge2) ==2:
        edge4 = origin_edge2[::-1]
        edge1 = origin_edge3[::-1]
        edge2 = origin_edge4[::-1]
        edge3 = origin_edge1[::-1]
    elif len(edge3) ==2:
        edge4 = origin_edge3
        edge3 = origin_edge2[::-1]
        edge2 = origin_edge3
        edge1 = origin_edge4[::-1]
    elif len(edge4) ==2:
        edge1 = origin_edge1
        edge2 = origin_edge2
        edge3 = origin_edge3
        edge4 = origin_edge4  
    else:
        aaa
    
    if len(edge1)==2:
        return build_unit_x_1_1_y(cdb_obj, edge1, edge2, edge3, edge4)
    if len(edge3)==2:
        return build_unit_x_1_1_y(cdb_obj, edge1, edge2, edge3, edge4)
    
    if len(edge1) > len(edge3):
        edge_l = edge1
        edge_s = edge3
        edge_m = edge2
    else:
        edge_l = edge3
        edge_s = edge1
        edge_m = edge2

    edge_index = 0
    pre_node = edge_s[0]
    for index in range(1, len(edge_s)):
        node = edge_s[index]
        
        next_edge_index = next_index(node, edge_l, edge_index, 1)
        temp_edge1 = [pre_node, node]
        if index == len(edge_s)-1:
            temp_edge2 = edge_m
        else:
            temp_edge2 = [node, edge_l[next_edge_index]]
        temp_edge3 = edge_l[edge_index:next_edge_index+1]
        temp_edge4 = [pre_node, edge_l[edge_index]]
        build_unit_x_1_1_y(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)
        
        edge_index = next_edge_index
        pre_node = node
        
    return origin_edge1, origin_edge2, origin_edge3, origin_edge4
        
    