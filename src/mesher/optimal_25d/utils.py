import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ref_lines = []
        
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise ValueError("index out of bound")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        raise ValueError("index out of bound")
        
    def set_line(self, line:'Line'):
        self.ref_lines.append(line)
        
    def get_lines(self):
        return self.ref_lines

class Line:
    def __init__(self, p1:Point, p2:Point):
        self.group = None
        self.p1 = p1
        self.p2 = p2
        self.axis = None
        self.second_axis = None
        
        p1.set_line(self)
        p2.set_line(self)
        
        if abs(p1[0] - p2[0]) <= 0.01 and abs(p1[1] - p2[1]) > 0.01:
            # vertical
            self.axis = 0
            self.second_axis = 1
        elif abs(p1[0] - p2[0]) > 0.01 and abs(p1[1] - p2[1]) <= 0.01:
            # horizon
            self.axis = 1
            self.second_axis = 0
        else:
            raise ValueError("Must be horizon or vertical")
        
    def is_group_alloable(self, line:'Line', size:float, angle:float = 45):
        if self.axis != line.axis:
            raise ValueError("Both lines should be same axis (horizon or vertical)")
        
        """
        Determines whether the test_line falls completely within the allowable region.
        
        Parameters:
        test_line (Line): test line
        angle (float): The taper angle A (in degrees).
        size (float): The maximum allowable horizontal width E on one side.
        
        Returns:
        bool: True if the test line is completely within the allowable region, False otherwise.
        """
        x_m, y_m1, y_m2 = self.p1[self.axis],  self.p1[self.second_axis],  self.p2[self.second_axis]
        x_t, y_t1, y_t2 = line.p1[self.axis],  line.p1[self.second_axis],  line.p2[self.second_axis]

        # Ensure Y coordinates are properly sorted (top is max, bottom is min)
        main_y_top = max(y_m1, y_m2)
        main_y_bot = min(y_m1, y_m2)
        test_y_top = max(y_t1, y_t2)
        test_y_bot = min(y_t1, y_t2)

        # 1. Check if the horizontal distance exceeds the maximum width E
        dx = abs(x_t - x_m)
        if dx > size:
            return False
            
        # If the test line is exactly on the main line (dx = 0), vertical clearance is 0
        if dx == 0:
            dy_clearance = 0
        else:
            # 2. Calculate the minimum vertical clearance (dy) caused by angle A
            # Based on the diagram, A is the angle from the vertical line, so tan(A) = dx / dy  =>  dy = dx / tan(A)
            if angle <= 0 or angle >= 90:
                raise ValueError("Angle A must be strictly between 0 and 90 degrees.")
                
            angle_rad = math.radians(angle)
            dy_clearance = dx / math.tan(angle_rad)

        # 3. Check if the test line is completely within the "top allowable zone"
        # The lowest point of the test line must be higher than or equal to (main top + vertical clearance)
        in_top_zone = (test_y_bot >= main_y_top + dy_clearance)

        # 4. Check if the test line is completely within the "bottom allowable zone"
        # The highest point of the test line must be lower than or equal to (main bottom - vertical clearance)
        in_bottom_zone = (test_y_top <= main_y_bot - dy_clearance)

        # The line is in the allowable region if it satisfies either the top or bottom zone condition
        return in_top_zone or in_bottom_zone
            
    def set_group(self, group:'LineGroup'):
        self.group = group
        
    def get_group(self):
        return self.group
        
class LineGroup:
    def __init__(self, size:float, angle:float=45, line:Line=None):
        self.lines = []
        self.size = size
        self.angle = angle
        
        if line is not None:
            self.lines.append(line)
        
    def is_group_alloable(self, line:Line):
        for line_g in self.lines:
            if not line_g.is_group_alloable(line, size=self.size, angle=self.angle):
                return False
        return True
    
    def add_line(self, line:Line):
        if not self.is_group_alloable(line):
            raise ValueError("Could not group line due to not independent")
        else:
            self.lines.append(line)
            line.set_group(self)
     
     
     
element_size = 5

points = []

groups_vertical = []
lines_vertical = [
    Line(Point(0,0),Point(0,10)),
    Line(Point(1,5),Point(1,15)),
    Line(Point(5,0),Point(5,10)),
    Line(Point(6,15),Point(6,19)),
    Line(Point(10,1),Point(10,11))
]

groups_horizon = []
lines_horizon = [
    Line(Point(0,0),Point(0,10)),
    Line(Point(1,5),Point(1,15)),
    Line(Point(5,0),Point(5,10)),
    Line(Point(6,15),Point(6,19)),
    Line(Point(10,1),Point(10,11))
]

# rule1 for vertical
for index, line in enumerate(lines_vertical):
    if index == 0 or index == len(lines_vertical)-1:
        group = LineGroup(element_size, line=line)
        groups_vertical.append(group)
    else:
        if groups_vertical[-1].is_group_alloable(line):
            groups_vertical[-1].add_line(line)
        else:
            group = LineGroup(element_size, line=line)
            groups_vertical.append(group)
            
# rule1 for horizon
for index, line in enumerate(lines_horizon):
    if index == 0 or index == len(lines_horizon)-1:
        group = LineGroup(element_size, line=line)
        groups_horizon.append(group)
    else:
        if groups_horizon[-1].is_group_alloable(line):
            groups_horizon[-1].add_line(line)
        else:
            group = LineGroup(element_size, line=line)
            groups_horizon.append(group)
            
# rule2: different node with same groups in both x and y
for point in points:
    ref_lines = point.get_lines()
    if len(ref_lines) > 1:
        
     
            
print(len(groups_vertical))
print(len(groups_horizon))