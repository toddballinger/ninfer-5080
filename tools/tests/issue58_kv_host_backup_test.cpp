#include "runtime/engine/issue58_kv_host_backup.h"
#include <cassert>
#include <type_traits>
using namespace ninfer::runtime::issue58;
static_assert(!std::is_copy_constructible_v<KvHostBackup>);
static_assert(std::is_nothrow_move_constructible_v<KvHostBackup>);
int main(){
    KvHostBackup backup;
    backup.append_plane({KvPageOrder::PageMajor,4,4096,1024,0,0},{2,0});
    backup.append_plane({KvPageOrder::HeadMajor,4,4096,128,512,8},{3,1});
    assert(backup.byte_count()==4096);
    assert(backup.planes().size()==2);
    assert(backup.planes()[0].image_offsets[0]==0);
    assert(backup.planes()[0].image_offsets[1]==1024);
    assert(backup.planes()[1].physical_page_ids[0]==3);
    assert(backup.planes()[1].data.size()==2048);
    KvHostBackup moved=std::move(backup);
    assert(moved.byte_count()==4096);
    const auto image=moved.page_image(0,1);
    assert(image && image->size()==1024);
    assert(!moved.page_image(0,2));
    assert(!moved.page_image(10,0));
    assert(moved.valid_restore_ids(std::vector<std::int32_t>{0,3}));
    assert(!moved.valid_restore_ids(std::vector<std::int32_t>{0,0}));
    assert(!moved.valid_restore_ids(std::vector<std::int32_t>{0,4}));
    assert(!moved.valid_restore_ids(std::vector<std::int32_t>{0}));
    bool rejected=false;
    try{moved.append_plane({KvPageOrder::PageMajor,4,4096,1024,0,0},{1,1});}
    catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
    rejected=false;
    try{moved.append_plane({KvPageOrder::PageMajor,4,4096,1024,0,0},{4});}
    catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
    assert(moved.byte_count()==4096);
}
