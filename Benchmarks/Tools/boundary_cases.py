"""Six original non-square fixtures with ties and reordered features."""


def cases():
    for device, dtype in [('cpu','float32'), ('gpu','float32'), ('cpu','float64')]:
        for layout, width, ascending in [('narrow',2,True), ('wide',5,False)]:
            names = [f'x{i}' for i in range(width)]
            # Float64 cases contain bits lost by Float32 conversion.
            epsilon = 2**-30 if dtype == 'float64' else 0
            p = {'operation':'dataframe-model','device':device,'dtype':dtype,
                 'row_ids':[91,17,63,28,45,39,72], 'feature_names':names, 'feature_order':list(reversed(names)),
                 'features':[[((r*7+c*3)%19-9)/8 + epsilon*(r+c+1) for c in range(width)] for r in range(7)],
                 'targets':[r/4-1+epsilon*r for r in range(7)], 'filter_values':[1,0,2,-1,1,3,0],
                 'filter_threshold':1, 'sort_keys':[2,7,1,0,1,2,3], 'ascending':ascending,
                 'weights':[(c+1)/8 for c in range(width)], 'bias':-.375+epsilon}
            yield f'boundary-{device}-{dtype}-{layout}', p
