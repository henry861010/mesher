def _get_new_edge2(element_size, edge1_len, edge2, edge3_len, edge4_len):
    ### calculate how many section at edge4 would be split
    target_section_num = max(round(distance(edge2[0], edge2[-1]) / element_size), 1)
    if (target_section_num + (edge1_len + edge3_len + edge4_len - 3)) % 2 == 1:
        if target_section_num == 1:
            target_section_num += 1
        else:
            target_section_num -= 1
            
    ### determine the outer node
    ### there are several reference node at the edge, we should consider it while we mesh.
    ### we call the distance between the reference node are "bucket"
    ###     * bucket_num: the number of the section in it
    ###     * bucket_dis: the distance of the bucket
    bucket_num = []
    bucket_dis = []
    smallest_dis = 1000000
    for i in range(1, len(edge2)):
        bucket_num.append(1)
        bucket_dis.append(distance(edge2[i-1], edge2[i]))
    
    for i in range(len(edge2)-1, target_section_num):
        ### find the bucket where each section in it have maximum len
        max_len = -1
        max_index = -1
        for j in range(len(bucket_dis)):
            temp_len = bucket_dis[j] / (bucket_num[j] + 1)
            if temp_len > max_len:
                max_len = temp_len
                max_index = j

        ### find it and add the section to this found bucket
        bucket_num[max_index] = bucket_num[max_index] + 1
    
    new_edge2 = []
    for i in range(0, len(bucket_num)):
        v_x = (edge2[i+1][0] - edge2[i][0]) / bucket_num[i]
        v_y = (edge2[i+1][1] - edge2[i][1]) / bucket_num[i]
        temp = [edge2[i]]
        for _ in range(bucket_num[i]):
            temp.append([temp[-1][0]+v_x, temp[-1][1]+v_y])
            
        smallest_dis = min(smallest_dis, bucket_dis[i]/bucket_num[i])

        new_edge2 = new_edge2[:-1] + temp
        
    return new_edge2, smallest_dis
        
def predict_wrap_line(criterias, element_size, left_num, section):
    num_list = []
    
    edge1 = section["edge1"]
    edge2 = section["edge2"]
    edge3 = section["edge3"]
    edge4 = section["edge4"]
    
    aspect_ration = criterias["aspect_ration"]
    minimum_progress_ration = criterias["minimum_progress_ration"]
    
    ### right_num == 2
    _, smallest_dis = _get_new_edge2(element_size, left_num, edge2, 2, len(edge4))
    if smallest_dis >= element_size * minimum_progress_ration:
        num_list.append(2)
        
    ### right_num == 3
    _, smallest_dis = _get_new_edge2(element_size, left_num, edge2, 3, len(edge4))
    if smallest_dis >= element_size * minimum_progress_ration:
        num_list.append(3)
        
    return num_list

def build_wrap_line(cdb_obj: 'Mesh2D', element_size, section):
    edge1 = section["edge1"]
    edge2 = section["edge2"]
    edge3 = section["edge3"]
    edge4 = section["edge4"]
    left_num = section["left_num"]
    right_num = section["right_num"]
    
    ### get the new edge2
    new_edge2, _ = _get_new_edge2(element_size, left_num, edge2, right_num, len(edge4))
    
    ### build the edge2
    left_index = 0
    left_outer_index = 0
    left_edge = edge1
    
    right_index = len(edge4) - 1
    right_outer_index = len(new_edge2) - 1
    right_edge = edge3
    
    count = 0
    while True:
        ### if finish get out from the while loop
        count += 1
        if count == len(new_edge2): 
            break
        
        ### build the left unit
        now_node = new_edge2[left_outer_index]
        next_node = new_edge2[left_outer_index + 1]
        if count == len(new_edge2)-1:
            next_left_index = right_index
        else:
            next_left_index = next_index(next_node, edge4, left_index, len(left_edge)-1, 1)
        
        temp_edge1 = left_edge
        temp_edge2 = [now_node, next_node]
        temp_edge3 = [edge4[next_left_index], next_node]
        temp_edge4 = edge4[left_index : next_left_index+1]
        
        ### build
        build_unit_x_1_1_y(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)

        ### update the temp var
        left_index = next_left_index
        left_outer_index = left_outer_index + 1
        left_edge = temp_edge3
        
        ### if finish get out from the while loop
        count += 1
        if count == len(new_edge2): 
            break
        
        ### build the right unit
        now_node = new_edge2[right_outer_index]
        next_node = new_edge2[right_outer_index - 1]
        if count == len(new_edge2)-1:
            next_right_index = left_index
        else:
            next_right_index = next_index(next_node, edge4, right_index, len(right_edge)-1, -1)
        
        temp_edge3 = right_edge
        temp_edge2 = [next_node, now_node]
        temp_edge1 = [edge4[next_right_index], next_node]
        temp_edge4 = edge4[next_right_index : right_index+1]
                
        ### build
        build_unit_x_1_1_y(cdb_obj, temp_edge1, temp_edge2, temp_edge3, temp_edge4)
        
        ### update the temp var
        right_index = next_right_index
        right_outer_index = right_outer_index - 1
        right_edge = temp_edge3
    return new_edge2
    