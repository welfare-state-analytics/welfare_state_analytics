def test_read_userlist_test():
    with open("userlist", "w") as fo:

        userlist = [ '# Users follows', "", "", "roger admin", "kalle ", "  kula", "  # no more users", ""]
        fo.write('\n'.join(userlist))

    with open("userlist", "r") as fi:
        lines = [ x.split() for x in [ y.strip() for y in fi.readlines() ] if len(x) > 0 and not x.startswith('#') ]

    users  = [ x[0] for x in lines ]
    admins = [ x[0] for x in lines if len(x) > 1 and x[1] == "admin" ]

    print(users, admins)
    