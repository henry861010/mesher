def predict_wrap_inner_left(criterias, element_size, left_num, section):
    edge2_len = len(section["edge2"])
    edge3_len = len(section["edge3"])
    edge4_len = len(section["edge4"])
    if (left_num + edge3_len + edge4_len) % 2 == 0:
        return [4]
    else:
        return [3]

def build_wrap_inner_left(cdb_obj: 'Mesh2D', element_size, section):
    edge1 = section["edge1"]
    edge2 = section["edge2"]
    edge3 = section["edge3"]
    edge4 = section["edge4"]
    left_num = section["left_num"]
    right_num = section["right_num"]
    
    if left_num == 3:
        edge1_center = [(edge1[0][0]+edge1[-1][0])/2, (edge1[0][1]+edge1[-1][1])/2]
        edge1 = [edge1[0], edge1_center, edge1[-1]]
    if right_num == 4:
        edge2_center = [(edge2[1][0]+edge2[-1][0])/2, (edge2[1][1]+edge2[-1][1])/2]
        edge2 = [edge2[0], edge2[1], edge2_center, edge2[3]]
    
    if left_num == 2 or len(edge4) == 2 or len(edge3) == 2:
        build_unit_x_y_z_1(cdb_obj, edge1, edge2, edge3, edge4)
    else:
        edge3_center_index = len(edge3)/2 + 1
        
        ### bottom
        temp_edge1 = [edge1[0], edge1[1]]
        temp_edge2 = [edge1[1], edge3[edge3_center_index]]
        temp_edge3 = edge3[:edge3_center_index]
        temp_edge4 = edge4
        build_unit_x_1_1_y(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)
        
        ### top
        temp_edge1 = [edge1[1], edge1[2]]
        temp_edge2 = edge2
        temp_edge3 = edge3[edge3_center_index:]
        temp_edge4 = [edge1[1], edge3[edge3_center_index]]
        build_unit_x_1_1_y(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)
        
    return [edge2[0], edge2[1]], []
    