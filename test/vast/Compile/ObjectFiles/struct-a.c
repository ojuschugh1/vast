// RUN: %vast-front -c -vast-use-pipeline=with-abi -o %t.vast.o %s && %clang -c -xc %s.driver -o %t.clang.o  && %clang %t.vast.o %t.clang.o -o %t && (%t; test $? -eq 0)

struct Data
{
    int x;
    int y;
};

int sum(struct Data d)
{
    return d.x + d.y;
}
