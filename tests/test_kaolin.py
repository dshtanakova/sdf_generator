import kaolin as kal


if __name__ == "__main__":
    mesh = kal.rep.TriangleMesh.from_obj('../data/0_simplified.obj')
    sdf = kal.conversions.trianglemesh_to_sdf(mesh)
    print(sdf)
