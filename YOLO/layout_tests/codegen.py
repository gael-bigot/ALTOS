import random
import os

loopvars = [("c_in", 1024), ("c_out", 1024), ("i", 13), ("j", 13), ("di", 3), ("dj", 3)]

data_layout_axis = [('H',13),('W',13),('C',1024)]
kernel_layout_axis = [('H',3), ('W',3),('C',1024), ('N',1024)]


src_template_start = """
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

void conv2d(float* x, float* w, float* y){
"""

scr_template_main = """
int main(){
    srand(time(NULL));
    float* x = aligned_alloc(32, 15*15*1024*sizeof(float));
    float* w = aligned_alloc(32, 3*3*1024*1024*sizeof(float));
    float* y = aligned_alloc(32, 13*13*1024*sizeof(float));

    for (int i = 0; i < 15*15*1024; i++){
        x[i] = (float) rand()/ (float) RAND_MAX;
    }
    for (int i = 0; i < 3*3*1024*1024; i++){
        w[i] = (float) rand()/ (float) RAND_MAX;
    }

    clock_t begin = clock();

    conv2d(x, w, y);

    clock_t end = clock();

    unsigned long millis = (end -  begin) * 1000 / CLOCKS_PER_SEC;

    printf( "Finished in %ld ms\\n", millis );

}
"""
while True:
    with open("source.c", "w") as f:
        f.write(src_template_start)
        random.shuffle(loopvars)
        #random.shuffle(data_layout_axis)
        #random.shuffle(kernel_layout_axis)

        if loopvars[-1][0] != "c_out":
            continue
        if data_layout_axis[-1][0] != "C":
            continue
        if kernel_layout_axis[-1][0] != "N":
            continue

        vars, _ = zip(*loopvars)
        loop_order = " ".join(vars)

        for var, span in loopvars:
            f.write(f"\tfor (int {var} = 0; {var} < {span}; {var}++)"+"{\n")

        axis, spans = zip(*data_layout_axis)
        axis2vars_x = {"H":"(i+di)", "W":"(j+dj)", "C": "c_in"}
        axis2vars_y = {"H":"i", "W":"j", "C": "c_out"}
        vars_x = [axis2vars_x[a] for a in axis]
        vars_y = [axis2vars_y[a] for a in axis]

        data_layout = "".join(axis)

        f.write(f"\t\tint x_index = {vars_x[0]}*{spans[1]}*{spans[2]} + {vars_x[1]}*{spans[2]} + {vars_x[2]};\n")
        f.write(f"\t\tint y_index = {vars_y[0]}*{spans[1]}*{spans[2]} + {vars_y[1]}*{spans[2]} + {vars_y[2]};\n")

        axis, spans = zip(*kernel_layout_axis)
        axis2vars = {"H":"di", "W":"dj", "N":"c_out", "C":"c_in"}
        vars = [axis2vars[a] for a in axis]

        kernel_layout = "".join(axis)

        f.write(f"\t\tint w_index = {vars[0]}*{spans[1]}*{spans[2]}*{spans[3]} + {vars[1]}*{spans[2]}*{spans[3]} + {vars[2]}*{spans[3]} + {vars[3]};\n")   
        f.write("\t\ty[y_index] += x[x_index] * w[w_index];\n\t}}}}}}\n}\n")

        f.write(scr_template_main)

    print(f"DATA LAYOUT : {data_layout}")
    print(f"KERNEL_LAYOUT : {kernel_layout}")
    print(f"LOOP ORDER : {loop_order}")

    os.system("gcc -o tmp source.c -O3")
    os.system("./tmp")