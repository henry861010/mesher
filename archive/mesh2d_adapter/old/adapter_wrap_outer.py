def predict_wrap_outer(criterias, element_size, left_num, section):
    ### for the outer corner, the number of nodes of both edge1 and edge4 should be equal
    return [left_num]

def build_wrap_outer(cdb_obj: 'Mesh2D', element_size, section):
    edge1 = section["edge1"]
    edge2 = section["edge2"]
    edge3 = section["edge3"]
    edge4 = section["edge4"]
    left_num = section["left_num"]
    right_num = section["right_num"]
    
    if left_num == 2 and right_num == 2:
        (edge1, edge2, edge3, edge4) = build_unit_x_1_1_y(cdb_obj, edge1, edge2, edge3, edge4)
    elif left_num == 2 and right_num == 2:
        edge1_center = [(edge1[0][0]+edge1[-1][0])/2, (edge1[0][1]+edge1[-1][1])/2]
        edge4_center = [(edge4[0][0]+edge4[-1][0])/2, (edge4[0][1]+edge4[-1][1])/2]
        edge1 = [edge1[0], edge1_center, edge1[-1]]
        edge4 = [edge4[0], edge4_center, edge4[-1]]
        (edge1, edge2, edge3, edge4) = build_unit_x_1_1_y(cdb_obj, edge1, edge2, edge3, edge4)
    return edge2, edge3[::-1]