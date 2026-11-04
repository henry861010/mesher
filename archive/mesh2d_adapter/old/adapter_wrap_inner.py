def predict_wrap_inner(criterias, element_size, left_num, section):
    edge1_len = left_num
    edge3_len = len(section["edge3"])
    edge4_len = len(section["edge4"])
    if (edge1_len + edge3_len + edge4_len) % 2 == 0:
        return [2]
    else:
        return [3]

def build_wrap_inner(cdb_obj: 'Mesh2D', element_size, section):
    edge1 = section["edge1"]
    edge2 = section["edge2"]
    edge3 = section["edge3"]
    edge4 = section["edge4"]
    left_num = section["left_num"]
    right_num = section["right_num"]
    
    if len(left_num)==2 and len(right_num)==2:
        (edge1, edge2, edge3, edge4) = build_unit_x_1_1_y(cdb_obj, edge1, edge2, edge3, edge4)
    elif len(left_num)==3 and len(right_num)==2:
        edge3_center_index = len(edge3)/2 + 1
        edge3_center = edge3[edge3_center_index]
        edge1_center = [(edge1[0][0]+edge1[-1][0])/2, (edge1[0][1]+edge1[-1][1])/2]
        
        bottom_edge1 = [edge1[0], edge1_center]
        bottom_edge2 = [edge1_center, edge3_center]
        bottom_edge3 = edge3[:edge3_center]
        bottom_edge4 = edge4
        build_unit_1_1_1_x(cdb_obj, bottom_edge1, bottom_edge2, bottom_edge3, bottom_edge4)
        
        top_edge1 = [edge1_center, edge1[-1]]
        top_edge2 = edge2
        top_edge3 = edge3[edge3_center:]
        top_edge4 = [edge1_center, edge3_center]
        build_unit_1_1_1_x(cdb_obj, top_edge1, top_edge2, top_edge3, top_edge4)
        
        edge1 = [edge1[0], edge1_center, edge1[-1]]
        edge2 = edge2
        edge3 = edge3
        edge4 = edge4
    elif len(left_num)==2 and len(right_num)==3:
        edge4_center_index = len(edge4)/2 + 1
        edge4_center = edge4[edge4_center_index]
        edge2_center = [(edge2[0][0]+edge2[-1][0])/2, (edge2[0][1]+edge2[-1][1])/2]
        
        left_edge1 = edge1
        left_edge2 = [edge2[0], edge2_center]
        left_edge3 = [edge4_center, edge2_center]
        left_edge4 = edge4[:edge4_center_index]
        build_unit_1_1_1_x(cdb_obj, left_edge1, left_edge2, left_edge3, left_edge4)
        
        right_edge1 = [edge4_center, edge2_center]
        right_edge2 = [edge2_center, edge2[-1]]
        right_edge3 = edge3
        right_edge4 = edge4[edge4_center_index:]
        build_unit_1_1_1_x(cdb_obj, right_edge1, right_edge2, right_edge3, right_edge4)
        
        edge1 = edge1
        edge2 = [edge2[0], edge2_center, edge2[-1]]
        edge3 = edge3
        edge4 = edge4
    elif len(left_num)==4 and len(right_num)==3:
        edge1_center = [(edge1[0][0]+edge1[-1][0])/2, (edge1[0][1]+edge1[-1][1])/2]
        edge2_center = [(edge2[0][0]+edge2[-1][0])/2, (edge2[0][1]+edge2[-1][1])/2]
        edge3_center_index = len(edge3)/2 + 1
        edge3_center = edge1[edge3_center_index]
        edge4_center_index = len(edge4)/2 + 1
        edge4_center = edge4[edge4_center_index]
        center = intersection([edge2_center, edge4_center], [edge1_center, edge3_center])
        
        leftbottom_edge1 = [edge1[0], edge1_center]
        leftbottom_edge2 = [edge1_center, center]
        leftbottom_edge3 = [edge4_center, center]
        leftbottom_edge4 = edge4[:edge4_center_index]
        build_unit_1_1_1_x(cdb_obj, leftbottom_edge1, leftbottom_edge2, leftbottom_edge3, leftbottom_edge4)
        
        lefttop_edge1 = [edge1_center, edge1[-1]]
        lefttop_edge2 = [edge1[-1], edge2_center]
        lefttop_edge3 = [center, edge2_center]
        lefttop_edge4 = [edge1_center, center]
        build_unit_1_1_1_x(cdb_obj, lefttop_edge1, lefttop_edge2, lefttop_edge3, lefttop_edge4)
        
        rightbottom_edge1 = [edge1_center, center]
        rightbottom_edge2 = [center, edge3_center]
        rightbottom_edge3 = [edge4[-1], edge3_center]
        rightbottom_edge4 = [edge4_center, edge4[-1]]
        build_unit_1_1_1_x(cdb_obj, rightbottom_edge1, rightbottom_edge2, rightbottom_edge3, rightbottom_edge4)
        
        righttop_edge1 = [center, edge2_center]
        righttop_edge2 = [edge2_center, edge2[-1]]
        righttop_edge3 = [edge3_center, edge2[-1]]
        righttop_edge4 = [center, edge3_center]
        build_unit_1_1_1_x(cdb_obj, righttop_edge1, righttop_edge2, righttop_edge3, righttop_edge4)
        
        edge1 = [edge1[0], edge1_center, edge1[-1]]
        edge2 = [edge2[0], edge2_center, edge2[-1]]
        edge3 = edge3
        edge4 = edge4
    return []