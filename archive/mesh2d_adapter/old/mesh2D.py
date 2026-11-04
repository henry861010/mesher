import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from my_math import *

import time
from test.test_util import *

class Mesh2D:
    def __init__(self):
        ### process
        self.element_2D = np.empty((0, 4), dtype=np.int32)
        self.node_2D = np.empty((0, 2), dtype=np.float32)
        
        ### others
        self.node_map = {}
        self.node_num = 0
        self.element_num = 0

    ### basic model building operation
    def add_node(self, x: float, y: float):
        node_key = f"{x}-{y}"
        if node_key not in self.node_map:
            new_node = np.array([[x, y]], dtype=np.float32)  # shape (1, 2)
            node_id = self.node_num
            self.node_2D[node_id] = new_node
            self.node_map[node_key] = node_id
            self.node_num += 1
            return (node_id, x, y)
        else:
            return (self.node_map[node_key], x, y)

    def add_element(self, node1_dim: list, node2_dim: list, node3_dim: list, node4_dim: list, comp_id: int = 0):
        node1_id, node1_x, node1_y = self.add_node(*node1_dim)
        node2_id, node2_x, node2_y = self.add_node(*node2_dim)
        node3_id, node3_x, node3_y = self.add_node(*node3_dim)
        node4_id, node4_x, node4_y = self.add_node(*node4_dim)

        element_id = len(self.element_2D)
        new_element = np.array([[node1_id, node2_id, node3_id, node4_id]], dtype=np.int32)  # shape (1, 10)

        element_id =  self.element_num
        self.element_2D[element_id] = new_element
        self.element_num += 1
        
        return element_id

    def pre_allocate_nodes(self, size: int = 1):
        '''
        Exponential growth (1.5x) is a standard amortized allocation method (like Python lists or C++ vectors).
        This avoids frequent resizing for small additions, keeping total reallocation count logarithmic in size.
        Keeps allocation tight when near expected final size.
        '''
        required = self.node_num + size
        current_capacity = len(self.node_2D)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            self.node_2D = np.vstack([self.node_2D, np.empty((extra, 2), dtype=np.float32)])

    def pre_allocate_elements(self, size: int = 1):
        '''
        Exponential growth (1.5x) is a standard amortized allocation method (like Python lists or C++ vectors).
        This avoids frequent resizing for small additions, keeping total reallocation count logarithmic in size.
        Keeps allocation tight when near expected final size.
        '''
        required = self.element_num + size
        current_capacity = len(self.element_2D)
        if required > current_capacity:
            new_capacity = max(required, int(current_capacity * 1.5))
            extra = new_capacity - current_capacity
            self.element_2D = np.vstack([self.element_2D, np.empty((extra, 4), dtype=np.int32)])
        
    ### wrap operation
    def build_block(self, element_size: float, x_list: list, y_list):
        new_x_list = [x_list[0]]
        new_y_list = [y_list[0]]
        
        ### calculate the reference point
        for i in range(1, len(x_list)):
            taget_num = math.ceil((x_list[i] - x_list[i-1]) / element_size)
            unit_len = (x_list[i] - x_list[i-1]) / taget_num
            now = x_list[i-1] + unit_len
            for _ in range(taget_num-1):
                new_x_list.append(now)
                now += unit_len
            new_x_list.append(x_list[i])
        for i in range(1, len(y_list)):
            taget_num = math.ceil((y_list[i] - y_list[i-1]) / element_size)
            unit_len = (y_list[i] - y_list[i-1]) / taget_num
            now = y_list[i-1] + unit_len
            for _ in range(taget_num-1):
                new_y_list.append(now)
                now += unit_len
            new_y_list.append(y_list[i])
            
        ### begin to build the checkerboard
        '''
        for i in range(1, len(new_x_list)):
            x_pre = new_x_list[i-1]
            x_post = new_x_list[i]
            for j in range(1, len(new_y_list)):
                y_pre = new_y_list[j-1]
                y_post = new_y_list[j]
                #print([x_pre, y_pre], [x_pre, y_post], [x_post, y_post], [x_post, y_pre])
                self.add_element([x_pre, y_pre], [x_pre, y_post], [x_post, y_post], [x_post, y_pre])      
        '''
        ### Build first (left), last (right), top, and bottom node lists
        coodinate_node_list = np.empty((len(new_y_list), 2), dtype=np.float32)

        # First column (left boundary)
        f_node_list = np.empty(len(new_y_list), dtype=np.int32)
        for i, y in enumerate(new_y_list):
            f_node_list[i] = self.add_node(new_x_list[0], y)
            coodinate_node_list[i] = [new_x_list[0], y]

        # Last column (right boundary)
        l_node_list = np.empty(len(new_y_list), dtype=np.int32)
        for i, y in enumerate(new_y_list):
            l_node_list[i] = self.add_node(new_x_list[-1], y)

        # Top row (top boundary)
        t_node_list = np.empty(len(new_x_list), dtype=np.int32)
        top_y = new_y_list[-1]
        for i, x in enumerate(new_x_list):
            t_node_list[i] = self.add_node(x, top_y)

        # Bottom row (bottom boundary)
        b_node_list = np.empty(len(new_x_list), dtype=np.int32)
        bottom_y = new_y_list[0]
        for i, x in enumerate(new_x_list):
            b_node_list[i] = self.add_node(x, bottom_y)

        ### Begin to build the blocks
        pre_node_list = f_node_list
        for index, x in enumerate(new_x_list[1:], start=1):  # Skip first and last
            if index < len(new_x_list)-1:
                # Build post_node_list
                post_node_list = np.empty(len(new_y_list), dtype=np.int32)

                # Interior nodes (excluding bottom and top)
                inner_count = len(new_y_list) - 2
                post_node_list[1:-1] = np.arange(inner_count, dtype=np.int32) + self.node_num

                # Set boundary node ids from bottom and top
                post_node_list[0] = b_node_list[index]
                post_node_list[-1] = t_node_list[index]

                # Set coordinates for interior nodes
                coodinate_node_list[1:-1, 0] = x
                coodinate_node_list[1:-1, 1] = new_y_list[1:-1]
                self.node_2D[self.node_num:self.node_num+inner_count] = coodinate_node_list[1:-1]
                self.node_num += inner_count
            else:
                post_node_list = l_node_list

            # Stack all elements at once
            elems = np.stack([
                pre_node_list[:-1],
                pre_node_list[1:],
                post_node_list[1:],
                post_node_list[:-1]
            ], axis=1)
            new_element_num = len(new_y_list) - 1
            self.element_2D[self.element_num:self.element_num+new_element_num] = elems
            self.element_num += new_element_num

            # Update for next iteration
            pre_node_list = post_node_list
            
    def build_blocks(self, element_size: float, section_type: list, table_x_dim: list, table_y_dim: list):
        ceil_table = [10000000 for _ in section_type]
        reference_node = [False for _ in table_x_dim]
        record_visit = [[False] * len(row) for row in section_type]
        
        section_num_x = len(table_x_dim)-1
        section_num_y = len(table_y_dim)-1
        
        for j in range(0, section_num_y):
            for i in range(0, section_num_x):
                if section_type[i][j] == 2 and not record_visit[i][j]:
                    ### find the rightest
                    right = i + 1
                    while right < section_num_x and section_type[right][j] == 2:
                        right += 1

                    ### find the ceil of each section (only execute while begin at y)
                    upper = 10000000
                    for m in range(i, right):
                        if j == 0 or section_type[m][j-1] != 2:
                            ### find the ceil
                            n = j + 1
                            while n < section_num_y and section_type[m][n] == 2:
                                n += 1
                            ceil_table[m] = n
                            
                        ### find the lowest ceil
                        upper = min(upper, ceil_table[m])
                        
                    ### left wall
                    if i > 1 and section_type[i-1][j] != 2:
                        n = j + 1
                        while n < section_num_y and section_type[i][n] != 2:
                            n += 1
                        upper = min(upper, n)
                    
                    ### right wall
                    if right != len(section_type) and section_type[right][j] != 2:
                        n = j + 1
                        while n < section_num_y and section_type[right][n] != 2:
                            n += 1
                        upper = min(upper, n)
                    
                    ### determine the reference element
                    for m in range(i+1, right-1):
                        ### clear the reference mark 
                        if section_type[m][j-1] != 2:
                            if section_type[m-1][j-1] != 2:
                                #reference_node[m] = False
                                print("~")
                            if section_type[m+1][j-1] != 2:
                                #reference_node[m+1] = False
                                print("~")
                        
                        ### set the reference node
                        if ceil_table[m] != ceil_table[m-1]:
                            reference_node[m] = True
                        if ceil_table[m] != ceil_table[m+1]:
                            reference_node[m+1] = True
                            
                        ### find the lowest ceil
                        upper = min(upper, ceil_table[m])
                        
                        ### update the reference mark
                        if m > i and ceil_table[m-1] != ceil_table[m]:
                            reference_node[m] = True
                            
                    reference_node[i] = True
                    reference_node[right] = True

                    ### set the visited section to true     
                    for m in range(i, right):
                        for n in range(j, upper):
                            record_visit[m][n] = True

                    ### build the x_list & y_list
                    y_list = [table_y_dim[j], table_y_dim[upper]]
                    x_list = []
                    for m in range(i, right+1):
                        if reference_node[m]:
                            x_list.append(table_x_dim[m])

                    ### build the block 
                    self.build_block(element_size, x_list, y_list)
                    print(f"{i} - {right}")
                    print(f"{j} - {upper}")
                    print(reference_node)
                    print(ceil_table)
                    print(x_list)
                    print("")
        
    ### CDB format
    def read_CDB(self):
        print("~")
        
    def write_cdb(self, path: str = 'cdb.txt', element_type: str = "185"):
        print("generate CDB")
            
    ### Debug
    def show_2d_graph(self, title: str = "2D mesh", pattern_lines = []):
        material_to_color = {
            index: cm.tab20(index / max(1, len(self.material_table) - 1)) for material, index in self.material_table.items()
        }
        for element in self.element_2D:
            x_coords = [
                self.node_2D[element[0]][0],
                self.node_2D[element[1]][0],
                self.node_2D[element[2]][0],
                self.node_2D[element[3]][0]
            ]
            y_coords = [
                self.node_2D[element[0]][1],
                self.node_2D[element[1]][1],
                self.node_2D[element[2]][1],
                self.node_2D[element[3]][1]
            ]
            
            plt.fill(x_coords, y_coords, edgecolor='black')
            
        for line in pattern_lines:
            x_vals = [line[0][0], line[-1][0]]
            y_vals = [line[0][1], line[-1][1]]
            plt.plot(x_vals, y_vals, color='blue', linewidth=1.5)

        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.title(f"{title}")
        plt.show()
        plt.close()
                
    def show_info(self):
        ### basic info
        print(f"[Basic Info]:")
        print(f"    total 2D elems: {len(self.element_2D)}")